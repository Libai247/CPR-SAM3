from collections import OrderedDict

import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.adaptive_branch_balance_sam3.segmentor import (
    AdaptiveBranchBalanceSegmentor,
    SCALE_SPECS,
)


MODE_SPECS = OrderedDict(
    [
        ('official', None),
        ('presence_safe', None),
        ('presence_extend', None),
        ('presence_vote', None),
        ('direct_raw', None),
        ('direct_safe', None),
        ('direct_extend', None),
        ('direct_vote', None),
        ('inverse_raw', None),
        ('inverse_safe', None),
        ('inverse_extend', None),
        ('inverse_vote', None),
        ('safe_majority', None),
        ('extend_majority', None),
        ('vote_majority', None),
        ('presence_supported_vote', None),
        ('branch_supported_vote', None),
        ('presence_supported_extend', None),
        ('unanimous_vote', None),
        ('agreement_weighted_raw', None),
        ('agreement_weighted_supported', None),
        ('route_s150_safe', None),
        ('route_s175_safe', None),
        ('route_s200_safe', None),
        ('route_s225_safe', None),
        ('route_s150_demand', None),
        ('route_s175_demand', None),
        ('route_s200_demand', None),
        ('route_s225_demand', None),
        ('hetero_h040', None),
        ('hetero_h060', None),
        ('hetero_h080', None),
        ('hetero_h100', None),
    ]
)
OUTPUT_MODES = tuple(MODE_SPECS)


@MODELS.register_module()
class MultiAxisConsensusSegmentor(AdaptiveBranchBalanceSegmentor):
    """Fuse independent presence and branch-balance prediction axes."""

    @staticmethod
    def _supported_correction(
        base,
        official,
        candidate,
        supports,
        require_all=False,
    ):
        selected = base.eq(official) & candidate.ne(official)
        agreements = [support.eq(candidate) for support in supports]
        support = agreements[0].clone()
        if require_all:
            for agreement in agreements[1:]:
                support &= agreement
        else:
            for agreement in agreements[1:]:
                support |= agreement
        output = base.clone()
        output[selected & support] = candidate[selected & support]
        return output

    @staticmethod
    def _agreement_weighted_logits(logits, predictions):
        first, second, third = predictions
        agree12 = first.eq(second).float().mean()
        agree13 = first.eq(third).float().mean()
        agree23 = second.eq(third).float().mean()
        weights = torch.stack(
            [
                agree12 + agree13,
                agree12 + agree23,
                agree13 + agree23,
            ]
        ).clamp_min(1e-6)
        weights = weights.to(
            device=logits[0].device,
            dtype=logits[0].dtype,
        )
        normalizer = weights.sum()
        return sum(
            weight * item
            for weight, item in zip(weights, logits)
        ) / normalizer

    @staticmethod
    def _support_count(candidate, predictions):
        return sum(
            prediction.eq(candidate).to(torch.int16)
            for prediction in predictions
        )

    @staticmethod
    def _route_direction(direct, inverse, strong, threshold):
        return inverse if strong >= float(threshold) else direct

    @staticmethod
    def _route_strength(candidates, demand):
        if demand < 1.50:
            return candidates[1]
        if demand < 2.50:
            return candidates[2]
        return candidates[3]

    def _build_multi_axis_predictions(
        self,
        semantic_query,
        instance_query,
        presence_query,
    ):
        unity_scale = semantic_query.new_ones(
            semantic_query.shape[0],
            1,
            1,
        )
        direct_scale = self._adaptive_scale(
            semantic_query,
            instance_query,
            presence_query,
            *SCALE_SPECS['top05_query'],
        )
        inverse_scale = self._adaptive_scale(
            semantic_query,
            instance_query,
            presence_query,
            *SCALE_SPECS['top05_query_inverse'],
        )

        presence_logits, presence = self._candidate_predictions(
            semantic_query,
            instance_query,
            presence_query,
            unity_scale,
        )
        direct_logits, direct = self._candidate_predictions(
            semantic_query,
            instance_query,
            presence_query,
            direct_scale,
        )
        inverse_logits, inverse = self._candidate_predictions(
            semantic_query,
            instance_query,
            presence_query,
            inverse_scale,
        )

        official = presence[0]
        safe_majority = self._majority(
            presence[1],
            direct[1],
            inverse[1],
            official,
        )
        extend_majority = self._majority(
            presence[2],
            direct[2],
            inverse[2],
            presence[1],
        )
        vote_majority = self._majority(
            presence[3],
            direct[3],
            inverse[3],
            presence[1],
        )
        presence_supported_vote = self._supported_correction(
            presence[1],
            official,
            presence[3],
            (direct[3], inverse[3]),
        )
        branch_supported_vote = self._supported_correction(
            presence[1],
            official,
            direct[3],
            (inverse[3],),
        )
        presence_supported_extend = self._supported_correction(
            presence[1],
            official,
            presence[2],
            (direct[2], inverse[2]),
        )
        unanimous_vote = self._supported_correction(
            presence[1],
            official,
            presence[3],
            (direct[3], inverse[3]),
            require_all=True,
        )

        vote_predictions = (
            presence[3],
            direct[3],
            inverse[3],
        )
        weighted_logits = self._agreement_weighted_logits(
            (
                presence_logits,
                direct_logits,
                inverse_logits,
            ),
            vote_predictions,
        )
        weighted_raw = self._official_prediction(weighted_logits)
        weighted_supported = presence[1].clone()
        support = self._support_count(
            weighted_raw,
            vote_predictions,
        )
        selected = (
            presence[1].eq(official)
            & weighted_raw.ne(official)
            & support.ge(2)
        )
        weighted_supported[selected] = weighted_raw[selected]

        official_changed = presence[0]
        semantic = self._aggregate_queries(semantic_query)
        presence_map = self._aggregate_queries(presence_query)
        (
            _,
            gamma075_query,
            gamma050_query,
        ) = self._scaled_outputs(
            semantic_query,
            instance_query,
            presence_query,
            unity_scale,
        )
        gamma075 = self._aggregate_queries(gamma075_query)
        gamma050 = self._aggregate_queries(gamma050_query)
        gamma075_pred = self._official_prediction(gamma075)
        gamma050_pred = self._official_prediction(gamma050)
        evidence075 = self._robust_evidence(
            semantic,
            presence_map.float().pow(0.75),
            0.99,
            use_presence=True,
        )
        evidence050 = self._robust_evidence(
            semantic,
            presence_map.float().pow(0.50),
            0.99,
            use_presence=True,
        )
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
        safe_changed = presence[1].ne(
            official_changed
        ).float().mean()
        vote_changed = presence[3].ne(
            official_changed
        ).float().mean()
        g075_changed = wide075.ne(
            official_changed
        ).float().mean()
        g050_changed = wide050.ne(
            official_changed
        ).float().mean()
        demand = float(
            (
                vote_changed
                / safe_changed.clamp_min(self.calibration_eps)
            ).item()
        )
        strong = float(
            (
                g050_changed
                / g075_changed.clamp_min(self.calibration_eps)
            ).item()
        )
        route_safe = []
        route_demand = []
        for threshold in (1.50, 1.75, 2.00, 2.25):
            selected_axis = self._route_direction(
                direct,
                inverse,
                strong,
                threshold,
            )
            route_safe.append(selected_axis[1])
            route_demand.append(
                self._route_strength(selected_axis, demand)
            )
        heterogeneity = float(
            (
                direct_scale.float().mean()
                - direct_scale.float().median()
            ).abs().item()
        )
        hetero_routes = []
        for threshold in (0.04, 0.06, 0.08, 0.10):
            if heterogeneity >= threshold:
                selected_axis = self._route_direction(
                    direct,
                    inverse,
                    strong,
                    2.25,
                )
                selected = self._route_strength(
                    selected_axis,
                    demand,
                )
            else:
                selected_axis = self._route_direction(
                    direct,
                    inverse,
                    strong,
                    1.50,
                )
                selected = selected_axis[1]
            hetero_routes.append(selected)

        predictions = torch.stack(
            [
                official,
                presence[1],
                presence[2],
                presence[3],
                direct[0],
                direct[1],
                direct[2],
                direct[3],
                inverse[0],
                inverse[1],
                inverse[2],
                inverse[3],
                safe_majority,
                extend_majority,
                vote_majority,
                presence_supported_vote,
                branch_supported_vote,
                presence_supported_extend,
                unanimous_vote,
                weighted_raw,
                weighted_supported,
                *route_safe,
                *route_demand,
                *hetero_routes,
            ]
        ).to(torch.int16)

        evidence = self._robust_evidence(
            semantic,
            presence_map,
            0.99,
            use_presence=True,
        )
        top2, semantic_choice, _ = self._top2_context(
            presence_logits,
            evidence,
        )
        diagnostics = torch.stack(
            [
                top2[0],
                top2[1],
                semantic_choice,
                direct[3],
                inverse[3],
            ]
        ).to(torch.int16)
        scale_summary = torch.stack(
            [
                direct_scale.new_tensor(strong).float(),
                direct_scale.new_tensor(demand).float(),
                direct_scale.new_tensor(heterogeneity).float(),
                safe_changed.float(),
                vote_changed.float(),
                g075_changed.float(),
                g050_changed.float(),
                direct_scale.float().mean(),
                direct_scale.float().median(),
                inverse_scale.float().mean(),
                inverse_scale.float().median(),
            ]
        )
        return (
            presence_logits,
            predictions,
            diagnostics,
            scale_summary,
        )

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
            logits, predictions, diagnostics, scale_summary = (
                self._build_multi_axis_predictions(
                    semantic_query,
                    instance_query,
                    presence_query,
                )
            )
            official = predictions[0].long()
            parity = official.ne(
                self._official_prediction(logits)
            ).float().amax()
            data_sample.set_data(
                {
                    'seg_logits': PixelData(data=logits),
                    'pred_sem_seg': PixelData(
                        data=official.unsqueeze(0)
                    ),
                    'multi_axis_preds': PixelData(data=predictions),
                    'multi_axis_diagnostics': PixelData(
                        data=diagnostics
                    ),
                    'multi_axis_parity': PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    'multi_axis_scale_summary': PixelData(
                        data=scale_summary.reshape(1, -1)
                    ),
                }
            )
        return data_samples
