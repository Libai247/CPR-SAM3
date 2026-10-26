import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.multi_axis_consensus_sam3.segmentor import (
    OUTPUT_MODES as BASE_OUTPUT_MODES,
)
from cpr_sam3.research.multi_axis_consensus_sam3.segmentor import (
    MultiAxisConsensusSegmentor,
)


OUTPUT_MODES = BASE_OUTPUT_MODES + ('rchf',)


@MODELS.register_module()
class HierarchicalEquivarianceResponseSegmentor(
    MultiAxisConsensusSegmentor
):
    """Export single-view response experts and the RCHF expert together."""

    @staticmethod
    def _rotate_90(image):
        transpose = getattr(Image, 'Transpose', Image)
        return image.transpose(transpose.ROTATE_90)

    @staticmethod
    def _undo_rotate_90(tensor):
        return torch.rot90(tensor, k=-1, dims=(-2, -1))

    def _single_view_outputs(self, image):
        if self.slide_crop > 0 and (
            self.slide_crop < image.size[0]
            or self.slide_crop < image.size[1]
        ):
            return self._slide_inference_branches(image)
        return self._inference_single_view_branches(image)

    def _resize_outputs(self, outputs, target_hw):
        return tuple(
            self._resize_map(
                output.unsqueeze(0),
                target_hw,
            ).squeeze(0)
            if output.shape[-2:] != target_hw
            else output
            for output in outputs
        )

    def _rchf_candidate(self, identity, aligned):
        identity_presence = (
            identity[3]
            if self.use_presence_score
            else torch.ones_like(identity[3])
        )
        aligned_presence = (
            aligned[3]
            if self.use_presence_score
            else torch.ones_like(aligned[3])
        )
        identity_semantic = self._aggregate_queries(
            identity[1] * identity_presence
        )
        identity_instance = self._aggregate_queries(
            identity[2] * identity_presence
        )
        aligned_semantic = self._aggregate_queries(
            aligned[1] * aligned_presence
        )
        aligned_instance = self._aggregate_queries(
            aligned[2] * aligned_presence
        )
        semantic_mean = (
            identity_semantic + aligned_semantic
        ) * 0.5
        instance_mean = (
            identity_instance + aligned_instance
        ) * 0.5
        semantic_difference = (
            identity_semantic - aligned_semantic
        ).abs()
        instance_difference = (
            identity_instance - aligned_instance
        ).abs()
        identity_fused = self._aggregate_queries(identity[0])
        aligned_fused = self._aggregate_queries(aligned[0])
        fused_difference = (
            identity_fused - aligned_fused
        ).abs()
        semantic_reliability = (
            1.0 - semantic_difference
        ).clamp(0.0, 1.0)
        instance_reliability = (
            1.0 - instance_difference
        ).clamp(0.0, 1.0)
        logits = torch.maximum(
            semantic_mean * semantic_reliability,
            instance_mean * instance_reliability,
        )
        stats = torch.stack(
            [
                semantic_difference.float().mean(),
                instance_difference.float().mean(),
                fused_difference.float().mean(),
            ]
        )
        return self._official_prediction(logits), stats

    def predict(self, inputs, data_samples):
        del inputs
        for sample in data_samples:
            meta = sample.metainfo
            image = Image.open(meta['img_path']).convert('RGB')
            target_hw = tuple(meta['ori_shape'][:2])
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
            official = base_predictions[0].long()
            parity = official.ne(
                self._official_prediction(logits)
            ).float().amax()
            rchf_scale = scale_summary[7]
            route_stats = torch.cat(
                [
                    scale_summary,
                    rchf_scale.reshape(1),
                    rchf_errors,
                ]
            )
            sample.set_data(
                {
                    'seg_logits': PixelData(data=logits),
                    'pred_sem_seg': PixelData(
                        data=official.unsqueeze(0)
                    ),
                    'hierarchical_candidate_preds': PixelData(
                        data=predictions
                    ),
                    'multi_axis_diagnostics': PixelData(
                        data=diagnostics
                    ),
                    'multi_axis_parity': PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    'multi_axis_scale_summary': PixelData(
                        data=scale_summary.reshape(1, -1)
                    ),
                    'hierarchical_route_stats': PixelData(
                        data=route_stats.reshape(1, -1)
                    ),
                }
            )
        return data_samples
