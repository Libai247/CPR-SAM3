from collections import OrderedDict

import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.branch_p3c_sam3.segmentor import BranchP3CSegmentor


MODE_SPECS = OrderedDict(
    [
        ("official", None),
        ("p3c_safe", None),
        ("p3c_wide", None),
        ("g075_wide", None),
        ("g050_wide", None),
        ("extend_g075", None),
        ("extend_g075_official", None),
        ("extend_all_temps", None),
        ("vote_all_temps", None),
    ]
)
OUTPUT_MODES = tuple(MODE_SPECS)


@MODELS.register_module()
class TempConsensusP3CSegmentor(BranchP3CSegmentor):
    """Extend safe P3C only with temperature-consistent corrections."""

    def _wide(self, logits, prediction, evidence):
        return self._calibrated_prediction(
            logits,
            prediction,
            evidence,
            weight=0.50,
            margin=0.50,
            top2_only=True,
        )

    @staticmethod
    def _extend(base, official, candidate, *guards):
        output = base.clone()
        selected = base.eq(official) & candidate.ne(official)
        for guard in guards:
            selected &= guard.eq(candidate)
        output[selected] = candidate[selected]
        return output

    @staticmethod
    def _majority(first, second, third, fallback):
        output = fallback.clone()
        first_second = first.eq(second)
        first_third = first.eq(third)
        second_third = second.eq(third)
        output[first_second] = first[first_second]
        output[~first_second & first_third] = first[
            ~first_second & first_third
        ]
        output[~first_second & ~first_third & second_third] = second[
            ~first_second & ~first_third & second_third
        ]
        return output

    def _build_consensus_predictions(
        self,
        official,
        semantic,
        presence,
        gamma075,
        gamma050,
    ):
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
        wide075 = self._wide(gamma075, gamma075_pred, evidence075)
        wide050 = self._wide(gamma050, gamma050_pred, evidence050)
        extend075 = self._extend(
            safe,
            official_pred,
            wide,
            wide075,
        )
        extend075_official = self._extend(
            safe,
            official_pred,
            wide,
            wide075,
            gamma075_pred,
        )
        extend_all = self._extend(
            safe,
            official_pred,
            wide,
            wide075,
            wide050,
        )
        vote = self._majority(
            wide,
            wide075,
            wide050,
            safe,
        )

        top2, semantic_choice, _ = self._top2_context(
            official,
            evidence,
        )
        _, choice075, _ = self._top2_context(
            gamma075,
            evidence075,
        )
        _, choice050, _ = self._top2_context(
            gamma050,
            evidence050,
        )
        diagnostics = torch.stack(
            [
                top2[0],
                top2[1],
                semantic_choice,
                choice075,
                choice050,
            ]
        ).to(torch.int16)
        predictions = torch.stack(
            [
                official_pred,
                safe,
                wide,
                wide075,
                wide050,
                extend075,
                extend075_official,
                extend_all,
                vote,
            ]
        ).to(torch.int16)
        return predictions, diagnostics

    def predict(self, inputs, data_samples):
        del inputs
        for data_sample in data_samples:
            meta = data_sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            if self.slide_crop > 0 and (
                self.slide_crop < image.size[0]
                or self.slide_crop < image.size[1]
            ):
                query_outputs = self._slide_inference_branches(image)
            else:
                query_outputs = self._inference_single_view_branches(image)

            ori_shape = tuple(meta["ori_shape"][:2])
            if query_outputs[0].shape[-2:] != ori_shape:
                query_outputs = tuple(
                    self._resize_map(
                        output.unsqueeze(0),
                        ori_shape,
                    ).squeeze(0)
                    for output in query_outputs
                )
            aggregated = tuple(
                self._aggregate_queries(output)
                for output in query_outputs
            )
            official, semantic, _, presence, gamma075, gamma050 = aggregated
            predictions, diagnostics = self._build_consensus_predictions(
                official,
                semantic,
                presence,
                gamma075,
                gamma050,
            )
            official_pred = predictions[0].long()
            parity = official_pred.ne(
                self._official_prediction(official)
            ).float().amax()
            data_sample.set_data(
                {
                    "seg_logits": PixelData(data=official),
                    "pred_sem_seg": PixelData(
                        data=official_pred.unsqueeze(0)
                    ),
                    "temp_consensus_preds": PixelData(data=predictions),
                    "temp_consensus_diagnostics": PixelData(
                        data=diagnostics
                    ),
                    "temp_consensus_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                }
            )
        return data_samples
