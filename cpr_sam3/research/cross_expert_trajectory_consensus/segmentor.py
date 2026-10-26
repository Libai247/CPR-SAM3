import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.unified_cross_foundation_trajectory.segmentor import (
    EXPERT_MODES,
    OUTPUT_MODES as UCTFS_OUTPUT_MODES,
    UnifiedCrossFoundationTrajectorySegmentor,
)


ROUTE_MODES = tuple(mode for mode in EXPERT_MODES if mode != "rchf")
VARIANTS = (
    "base",
    "rchf",
    "evidence_max",
    "base_switch_005",
    "base_switch_010",
    "base_switch_020",
    "rchf_switch_005",
    "rchf_switch_010",
    "rchf_switch_020",
)
OUTPUT_MODES = ("official",) + tuple(
    f"{variant}_{mode}"
    for variant in VARIANTS
    for mode in ROUTE_MODES
)


@MODELS.register_module()
class CrossExpertTrajectorySegmentor(
    UnifiedCrossFoundationTrajectorySegmentor
):
    """Resolve base/RCHF trajectory conflicts with frozen DINO evidence."""

    @staticmethod
    def _gather_score(score, prediction):
        return score.gather(0, prediction.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _switch(default, alternative, advantage, threshold):
        output = default.clone()
        selected = default.ne(alternative) & advantage.gt(float(threshold))
        output[selected] = alternative[selected]
        return output

    @torch.inference_mode()
    def predict(self, inputs, data_samples):
        data_samples = super().predict(inputs, data_samples)
        parent_indices = {
            mode: index for index, mode in enumerate(UCTFS_OUTPUT_MODES)
        }
        for sample in data_samples:
            meta = sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            target_hw = tuple(meta["ori_shape"][:2])
            parent = sample.uctfs_candidate_preds.data.long()
            official = parent[0]
            features = self.dino_extractor.extract(image)["last"]
            similarity = self._prototype_similarity(features, target_hw)
            support = self._spatial_support(similarity)
            evidence = similarity + 0.05 * support
            rchf = parent[
                parent_indices["cfv_trajectory_add_loose_rchf"]
            ]
            rchf_score = self._gather_score(evidence, rchf)
            candidates = {"official": official}
            for mode in ROUTE_MODES:
                base = parent[
                    parent_indices[
                        f"cfv_trajectory_add_loose_{mode}"
                    ]
                ]
                base_score = self._gather_score(evidence, base)
                base_advantage = base_score - rchf_score
                rchf_advantage = -base_advantage
                evidence_max = base.clone()
                choose_rchf = base.ne(rchf) & rchf_advantage.gt(0.0)
                evidence_max[choose_rchf] = rchf[choose_rchf]
                candidates[("base", mode)] = base
                candidates[("rchf", mode)] = rchf
                candidates[("evidence_max", mode)] = evidence_max
                for suffix, threshold in (
                    ("005", 0.005),
                    ("010", 0.010),
                    ("020", 0.020),
                ):
                    candidates[(f"base_switch_{suffix}", mode)] = (
                        self._switch(
                            base,
                            rchf,
                            rchf_advantage,
                            threshold,
                        )
                    )
                    candidates[(f"rchf_switch_{suffix}", mode)] = (
                        self._switch(
                            rchf,
                            base,
                            base_advantage,
                            threshold,
                        )
                    )

            predictions = [official]
            for variant in VARIANTS:
                for mode in ROUTE_MODES:
                    predictions.append(candidates[(variant, mode)])
            predictions = torch.stack(predictions).to(torch.int16)
            sample.set_data(
                {
                    "cetc_candidate_preds": PixelData(data=predictions),
                    "cetc_parity": PixelData(
                        data=sample.uctfs_parity.data.clone()
                    ),
                    "cetc_route_stats": PixelData(
                        data=sample.uctfs_route_stats.data.clone()
                    ),
                }
            )
        return data_samples
