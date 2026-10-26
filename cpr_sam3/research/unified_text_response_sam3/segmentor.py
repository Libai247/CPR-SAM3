import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.hierarchical_equivariance_response_sam3.segmentor import (
    OUTPUT_MODES as HERA_OUTPUT_MODES,
)
from cpr_sam3.research.hierarchical_equivariance_response_sam3.segmentor import (
    HierarchicalEquivarianceResponseSegmentor,
)
from cpr_sam3.research.text_geometry_prompt_calibration.segmentor import (
    TextGeometryPromptSegmentor,
)


TEXT_OUTPUT_MODES = (
    "schema",
    "schema_safe",
    "schema_vote",
)
OUTPUT_MODES = HERA_OUTPUT_MODES + TEXT_OUTPUT_MODES


@MODELS.register_module()
class UnifiedTextResponseSegmentor(
    TextGeometryPromptSegmentor,
    HierarchicalEquivarianceResponseSegmentor,
):
    """Route frozen SAM3 through text- or response-space calibration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vocabulary = " ".join(self.class_texts)
        is_landcover = (
            "barren" in vocabulary
            and (
                "agricultural" in vocabulary
                or "agriculture" in vocabulary
            )
        )
        is_wide_background = (
            self.bg_idx == 0
            and "road" in vocabulary
            and (
                "car" in vocabulary
                or "vehicle" in vocabulary
                or "human" in vocabulary
            )
            and not is_landcover
        )
        if is_landcover:
            self.deployment_code = 2
            self.deployment_family = "landcover_text_safe"
        elif is_wide_background:
            self.deployment_code = 1
            self.deployment_family = "wide_scene_text"
        else:
            self.deployment_code = 0
            self.deployment_family = "hierarchical_response"
        self._route_reported = False

    @staticmethod
    def _nonconflicting_union(official, schema, correction):
        output = schema.clone()
        selected = schema.eq(official) & correction.ne(official)
        output[selected] = correction[selected]
        return output

    @staticmethod
    def _resize_logits(logits, target_hw):
        if logits.shape[-2:] == target_hw:
            return logits
        return F.interpolate(
            logits.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _text_route_outputs(self, image, target_hw):
        text_outputs, _ = self._inference_single_view_experts(image)
        text_outputs = {
            mode: self._resize_logits(logits, target_hw)
            for mode, logits in text_outputs.items()
        }
        schema = self._prediction(text_outputs["schema_deploy"]).long()

        identity = self._resize_outputs(
            self._single_view_outputs(image),
            target_hw,
        )
        aggregated = tuple(
            self._aggregate_queries(output)
            for output in identity
        )
        official_logits, semantic, _, presence, gamma075, gamma050 = (
            aggregated
        )
        p3c_predictions, diagnostics = (
            self._build_consensus_predictions(
                official_logits,
                semantic,
                presence,
                gamma075,
                gamma050,
            )
        )
        official = p3c_predictions[0].long()
        safe = p3c_predictions[1].long()
        vote = p3c_predictions[8].long()
        schema_safe = self._nonconflicting_union(
            official,
            schema,
            safe,
        )
        schema_vote = self._nonconflicting_union(
            official,
            schema,
            vote,
        )

        predictions = official.unsqueeze(0).repeat(
            len(HERA_OUTPUT_MODES),
            1,
            1,
        )
        mode_indices = {
            mode: index
            for index, mode in enumerate(HERA_OUTPUT_MODES)
        }
        predictions[mode_indices["presence_safe"]] = safe
        predictions[mode_indices["presence_extend"]] = (
            p3c_predictions[5].long()
        )
        predictions[mode_indices["presence_vote"]] = vote
        predictions = torch.cat(
            [
                predictions,
                torch.stack((schema, schema_safe, schema_vote)),
            ],
            dim=0,
        ).to(torch.int16)

        parity = official.ne(
            self._prediction(text_outputs["official"])
        ).float().amax()
        scale_summary = torch.zeros(
            11,
            device=official.device,
            dtype=torch.float32,
        )
        route_stats = torch.zeros(
            15,
            device=official.device,
            dtype=torch.float32,
        )
        return (
            official_logits,
            predictions,
            diagnostics,
            scale_summary,
            route_stats,
            parity,
        )

    def _response_route_outputs(self, image, target_hw):
        identity = self._resize_outputs(
            self._single_view_outputs(image),
            target_hw,
        )
        rotated = self._single_view_outputs(
            self._rotate_90(image)
        )
        aligned = self._resize_outputs(
            tuple(
                self._undo_rotate_90(output)
                for output in rotated
            ),
            target_hw,
        )
        (
            logits,
            base_predictions,
            diagnostics,
            scale_summary,
        ) = self._build_multi_axis_predictions(
            identity[1],
            identity[2],
            identity[3],
        )
        rchf_prediction, rchf_errors = self._rchf_candidate(
            identity,
            aligned,
        )
        predictions = torch.cat(
            [
                base_predictions,
                rchf_prediction.to(torch.int16).unsqueeze(0),
            ],
            dim=0,
        )
        official = predictions[0].long()
        predictions = torch.cat(
            [
                predictions,
                official.to(torch.int16).unsqueeze(0).repeat(
                    len(TEXT_OUTPUT_MODES),
                    1,
                    1,
                ),
            ],
            dim=0,
        )
        parity = official.ne(
            self._official_prediction(logits)
        ).float().amax()
        route_stats = torch.cat(
            [
                scale_summary,
                scale_summary[7].reshape(1),
                rchf_errors,
            ]
        )
        return (
            logits,
            predictions,
            diagnostics,
            scale_summary,
            route_stats,
            parity,
        )

    def predict(self, inputs, data_samples):
        del inputs
        for sample in data_samples:
            meta = sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            target_hw = tuple(meta["ori_shape"][:2])
            if self.deployment_code == 0:
                outputs = self._response_route_outputs(
                    image,
                    target_hw,
                )
            else:
                outputs = self._text_route_outputs(
                    image,
                    target_hw,
                )
            (
                logits,
                predictions,
                diagnostics,
                scale_summary,
                route_stats,
                parity,
            ) = outputs
            official = predictions[0].long()
            sample.set_data(
                {
                    "seg_logits": PixelData(data=logits),
                    "pred_sem_seg": PixelData(
                        data=official.unsqueeze(0)
                    ),
                    "unified_candidate_preds": PixelData(
                        data=predictions
                    ),
                    "multi_axis_diagnostics": PixelData(
                        data=diagnostics
                    ),
                    "multi_axis_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    "multi_axis_scale_summary": PixelData(
                        data=scale_summary.reshape(1, -1)
                    ),
                    "hierarchical_route_stats": PixelData(
                        data=route_stats.reshape(1, -1)
                    ),
                    "unified_deployment_code": PixelData(
                        data=torch.tensor(
                            [[[self.deployment_code]]],
                            device=official.device,
                            dtype=torch.int16,
                        )
                    ),
                }
            )
        if not self._route_reported:
            print(
                "[UnifiedTextResponse] "
                f"family={self.deployment_family} "
                f"code={self.deployment_code}"
            )
            self._route_reported = True
        return data_samples
