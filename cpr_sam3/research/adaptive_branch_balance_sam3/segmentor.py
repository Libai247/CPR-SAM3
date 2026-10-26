from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.temp_consensus_p3c_sam3.segmentor import (
    TempConsensusP3CSegmentor,
)


SCALE_SPECS = OrderedDict(
    [
        (
            'top05_query',
            (0.05, False, 1.0, False, 0.50, 1.50),
        ),
        (
            'top05_query_narrow',
            (0.05, False, 1.0, False, 0.75, 1.25),
        ),
        (
            'top05_query_inverse',
            (0.05, False, 1.0, True, 0.50, 1.50),
        ),
    ]
)
CONSENSUS_SUFFIXES = ('raw', 'safe', 'extend', 'vote')
OUTPUT_MODES = ('official',) + tuple(
    f'{name}_{suffix}'
    for name in SCALE_SPECS
    for suffix in CONSENSUS_SUFFIXES
)


@MODELS.register_module()
class AdaptiveBranchBalanceSegmentor(TempConsensusP3CSegmentor):
    """Align semantic and instance response energy before OV3 fusion."""

    branch_scale_min = 0.50
    branch_scale_max = 1.50
    branch_stat_size = 128

    def __init__(self, *args, deploy_mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deploy_mode = deploy_mode
        valid_modes = set(OUTPUT_MODES) | {'vote_all_temps'}
        if deploy_mode is not None and deploy_mode not in valid_modes:
            raise ValueError(
                f'Unknown deploy_mode {deploy_mode!r}; '
                f'expected one of {sorted(valid_modes)}'
            )

    def _top_energy(self, branch, fraction):
        pooled = F.adaptive_avg_pool2d(
            branch.float().clamp_min(0).unsqueeze(1),
            (self.branch_stat_size, self.branch_stat_size),
        ).squeeze(1)
        flattened = pooled.flatten(1)
        count = max(1, int(round(flattened.shape[1] * fraction)))
        return flattened.topk(count, dim=1).values.mean(dim=1)

    def _adaptive_scale(
        self,
        semantic_query,
        instance_query,
        presence_query,
        fraction,
        global_scale,
        strength,
        inverse,
        scale_min,
        scale_max,
    ):
        semantic_energy = self._top_energy(semantic_query, fraction)
        instance_energy = self._top_energy(instance_query, fraction)
        valid = (semantic_energy > self.calibration_eps) & (
            instance_energy > self.calibration_eps
        )
        ratio = torch.ones_like(semantic_energy)
        if inverse:
            ratio[valid] = (
                instance_energy[valid] / semantic_energy[valid]
            )
        else:
            ratio[valid] = (
                semantic_energy[valid] / instance_energy[valid]
            )
        ratio = ratio.clamp(
            float(scale_min),
            float(scale_max),
        )

        if global_scale:
            present = presence_query[:, 0, 0].float() > 0.05
            selected = ratio[valid & present]
            if selected.numel() == 0:
                selected = ratio[valid]
            scalar = (
                selected.median()
                if selected.numel() > 0
                else ratio.new_tensor(1.0)
            )
            ratio = scalar.expand_as(ratio)

        if strength != 1.0:
            ratio = ratio.clamp_min(self.calibration_eps).pow(
                float(strength)
            )
        return ratio.view(-1, 1, 1)

    def _scaled_outputs(
        self,
        semantic_query,
        instance_query,
        presence_query,
        scale,
    ):
        raw = torch.maximum(semantic_query, instance_query * scale)
        if self.use_presence_score:
            official = raw * presence_query
            gamma075 = raw * presence_query.float().pow(0.75)
            gamma050 = raw * presence_query.float().pow(0.50)
        else:
            official = raw
            gamma075 = raw
            gamma050 = raw
        return official, gamma075, gamma050

    def _candidate_predictions(
        self,
        semantic_query,
        instance_query,
        presence_query,
        scale,
    ):
        official_query, gamma075_query, gamma050_query = (
            self._scaled_outputs(
                semantic_query,
                instance_query,
                presence_query,
                scale,
            )
        )
        official = self._aggregate_queries(official_query)
        gamma075 = self._aggregate_queries(gamma075_query)
        gamma050 = self._aggregate_queries(gamma050_query)
        semantic = self._aggregate_queries(semantic_query)
        presence = self._aggregate_queries(presence_query)

        official_pred = self._official_prediction(official)
        gamma075_pred = self._official_prediction(gamma075)
        gamma050_pred = self._official_prediction(gamma050)
        evidence = self._robust_evidence(
            semantic,
            presence,
            0.99,
            use_presence=True,
        )
        evidence075 = self._robust_evidence(
            semantic,
            presence.float().pow(0.75),
            0.99,
            use_presence=True,
        )
        evidence050 = self._robust_evidence(
            semantic,
            presence.float().pow(0.50),
            0.99,
            use_presence=True,
        )
        safe = self._calibrated_prediction(
            official,
            official_pred,
            evidence,
            weight=0.75,
            margin=0.20,
            top2_only=True,
        )
        wide = self._wide(official, official_pred, evidence)
        wide075 = self._wide(
            gamma075,
            gamma075_pred,
            evidence075,
        )
        wide050 = self._wide(
            gamma050,
            gamma050_pred,
            evidence050,
        )
        extend = self._extend(
            safe,
            official_pred,
            wide,
            wide075,
        )
        vote = self._majority(
            wide,
            wide075,
            wide050,
            safe,
        )
        return official, (official_pred, safe, extend, vote)

    def _build_adaptive_predictions(
        self,
        semantic_query,
        instance_query,
        presence_query,
    ):
        base_query = torch.maximum(semantic_query, instance_query)
        if self.use_presence_score:
            base_query = base_query * presence_query
        base = self._aggregate_queries(base_query)
        official_pred = self._official_prediction(base)
        predictions = [official_pred]
        scale_stats = []

        for (
            fraction,
            global_scale,
            strength,
            inverse,
            scale_min,
            scale_max,
        ) in SCALE_SPECS.values():
            scale = self._adaptive_scale(
                semantic_query,
                instance_query,
                presence_query,
                fraction,
                global_scale,
                strength,
                inverse,
                scale_min,
                scale_max,
            )
            _, candidate_predictions = self._candidate_predictions(
                semantic_query,
                instance_query,
                presence_query,
                scale,
            )
            predictions.extend(candidate_predictions)
            scale_values = scale.flatten().float()
            scale_stats.append(
                torch.stack(
                    [
                        scale_values.mean(),
                        scale_values.median(),
                        scale_values.min(),
                        scale_values.max(),
                    ]
                )
            )

        return (
            base,
            torch.stack(predictions).to(torch.int16),
            torch.stack(scale_stats),
        )

    def _build_deploy_prediction(
        self,
        semantic_query,
        instance_query,
        presence_query,
    ):
        if self.deploy_mode == 'official':
            scale = semantic_query.new_ones(
                semantic_query.shape[0],
                1,
                1,
            )
            official, predictions = self._candidate_predictions(
                semantic_query,
                instance_query,
                presence_query,
                scale,
            )
            return official, predictions[0]

        if self.deploy_mode == 'vote_all_temps':
            scale = semantic_query.new_ones(
                semantic_query.shape[0],
                1,
                1,
            )
            official, predictions = self._candidate_predictions(
                semantic_query,
                instance_query,
                presence_query,
                scale,
            )
            return official, predictions[3]

        estimator = next(
            (
                name
                for name in sorted(
                    SCALE_SPECS,
                    key=len,
                    reverse=True,
                )
                if self.deploy_mode.startswith(f'{name}_')
            ),
            None,
        )
        if estimator is None:
            raise RuntimeError(
                f'Cannot parse deploy_mode {self.deploy_mode!r}'
            )
        suffix = self.deploy_mode[len(estimator) + 1 :]
        scale = self._adaptive_scale(
            semantic_query,
            instance_query,
            presence_query,
            *SCALE_SPECS[estimator],
        )
        return self._selected_candidate_prediction(
            semantic_query,
            instance_query,
            presence_query,
            scale,
            suffix,
        )

    def _selected_candidate_prediction(
        self,
        semantic_query,
        instance_query,
        presence_query,
        scale,
        suffix,
    ):
        official_query, gamma075_query, gamma050_query = (
            self._scaled_outputs(
                semantic_query,
                instance_query,
                presence_query,
                scale,
            )
        )
        official = self._aggregate_queries(official_query)
        official_pred = self._official_prediction(official)
        if suffix == 'raw':
            return official, official_pred

        semantic = self._aggregate_queries(semantic_query)
        presence = self._aggregate_queries(presence_query)
        evidence = self._robust_evidence(
            semantic,
            presence,
            0.99,
            use_presence=True,
        )
        safe = self._calibrated_prediction(
            official,
            official_pred,
            evidence,
            weight=0.75,
            margin=0.20,
            top2_only=True,
        )
        if suffix == 'safe':
            return official, safe

        gamma075 = self._aggregate_queries(gamma075_query)
        gamma075_pred = self._official_prediction(gamma075)
        evidence075 = self._robust_evidence(
            semantic,
            presence.float().pow(0.75),
            0.99,
            use_presence=True,
        )
        wide = self._wide(official, official_pred, evidence)
        wide075 = self._wide(
            gamma075,
            gamma075_pred,
            evidence075,
        )
        extend = self._extend(
            safe,
            official_pred,
            wide,
            wide075,
        )
        if suffix == 'extend':
            return official, extend

        gamma050 = self._aggregate_queries(gamma050_query)
        gamma050_pred = self._official_prediction(gamma050)
        evidence050 = self._robust_evidence(
            semantic,
            presence.float().pow(0.50),
            0.99,
            use_presence=True,
        )
        wide050 = self._wide(
            gamma050,
            gamma050_pred,
            evidence050,
        )
        vote = self._majority(
            wide,
            wide075,
            wide050,
            safe,
        )
        return official, vote

    def predict(self, inputs, data_samples):
        del inputs
        for data_sample in data_samples:
            meta = data_sample.metainfo
            image = Image.open(meta['img_path']).convert('RGB')
            if self.slide_crop > 0 and (
                self.slide_crop < image.size[0]
                or self.slide_crop < image.size[1]
            ):
                query_outputs = self._slide_inference_branches(image)
            else:
                query_outputs = self._inference_single_view_branches(
                    image
                )

            ori_shape = tuple(meta['ori_shape'][:2])
            if query_outputs[0].shape[-2:] != ori_shape:
                query_outputs = tuple(
                    self._resize_map(
                        output.unsqueeze(0),
                        ori_shape,
                    ).squeeze(0)
                    for output in query_outputs
                )

            _, semantic_query, instance_query, presence_query, _, _ = (
                query_outputs
            )
            if self.deploy_mode is not None:
                official, selected = self._build_deploy_prediction(
                    semantic_query,
                    instance_query,
                    presence_query,
                )
                data_sample.set_data(
                    {
                        'seg_logits': PixelData(data=official),
                        'pred_sem_seg': PixelData(
                            data=selected.long().unsqueeze(0)
                        ),
                    }
                )
                continue

            official, predictions, scale_stats = (
                self._build_adaptive_predictions(
                    semantic_query,
                    instance_query,
                    presence_query,
                )
            )
            official_pred = predictions[0].long()
            parity = official_pred.ne(
                self._official_prediction(official)
            ).float().amax()
            data_sample.set_data(
                {
                    'seg_logits': PixelData(data=official),
                    'pred_sem_seg': PixelData(
                        data=official_pred.unsqueeze(0)
                    ),
                    'branch_calibration_preds': PixelData(
                        data=predictions
                    ),
                    'branch_calibration_diagnostics': PixelData(
                        data=torch.stack(
                            [official_pred] * 5
                        ).to(torch.int16)
                    ),
                    'branch_calibration_parity': PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    'adaptive_branch_scale_stats': PixelData(
                        data=scale_stats
                    ),
                }
            )
        return data_samples
