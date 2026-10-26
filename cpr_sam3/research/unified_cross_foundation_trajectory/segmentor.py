from pathlib import Path

import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.cross_foundation_candidate_verifier.segmentor import (
    CrossFoundationCandidateVerifierSegmentor,
)
from cpr_sam3.research.multi_axis_consensus_sam3.segmentor import (
    OUTPUT_MODES as HERA_BASE_MODES,
)
from cpr_sam3.research.sam3_arch_probe.probe import SAM3ProbeCapture
from cpr_sam3.research.satellite_affinity_sam3.dino import DINOv3DenseExtractor
from cpr_sam3.research.trajectory_cross_foundation_selector.segmentor import (
    TrajectoryCrossFoundationSegmentor,
)
from cpr_sam3.research.unified_text_response_sam3.segmentor import (
    UnifiedTextResponseSegmentor,
)


HERA_EXPERT_MODES = (
    "presence_safe",
    "presence_vote",
    "direct_raw",
    "direct_safe",
    "direct_vote",
    "inverse_safe",
    "inverse_extend",
    "rchf",
)
EXPERT_MODES = HERA_EXPERT_MODES + ("schema_route",)
POLICIES = (
    "raw",
    "cfv",
    "trajectory_balanced",
    "trajectory_loose",
    "cfv_trajectory_balanced",
    "cfv_trajectory_loose",
    "cfv_trajectory_add_balanced",
    "cfv_trajectory_add_loose",
)
OUTPUT_MODES = ("official",) + tuple(
    f"{policy}_{mode}"
    for policy in POLICIES
    for mode in EXPERT_MODES
)


@MODELS.register_module()
class UnifiedCrossFoundationTrajectorySegmentor(
    UnifiedTextResponseSegmentor
):
    """Verify response/text candidates with DINO and decoder trajectories."""

    _load_prototype_bank = (
        CrossFoundationCandidateVerifierSegmentor._load_prototype_bank
    )
    _prototype_similarity = (
        CrossFoundationCandidateVerifierSegmentor._prototype_similarity
    )
    _spatial_support = staticmethod(
        CrossFoundationCandidateVerifierSegmentor._spatial_support
    )
    _verify_candidate = (
        CrossFoundationCandidateVerifierSegmentor._verify_candidate
    )
    _prototype_rescue = (
        CrossFoundationCandidateVerifierSegmentor._prototype_rescue
    )
    _layer_query_logits = (
        TrajectoryCrossFoundationSegmentor._layer_query_logits
    )
    _inference_view_with_layers = (
        TrajectoryCrossFoundationSegmentor._inference_view_with_layers
    )
    _slide_view_with_layers = (
        TrajectoryCrossFoundationSegmentor._slide_view_with_layers
    )
    _view_with_layers = TrajectoryCrossFoundationSegmentor._view_with_layers
    _resize_view = TrajectoryCrossFoundationSegmentor._resize_view
    _trajectory_candidate = (
        TrajectoryCrossFoundationSegmentor._trajectory_candidate
    )
    _aggregate_layer_queries = (
        TrajectoryCrossFoundationSegmentor._aggregate_layer_queries
    )

    def __init__(
        self,
        *args,
        prototype_bank_path=None,
        dino_weights=(
            "checkpoints/"
            "dinov3_server_bundle/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
        dino_domain="lvd",
        dino_max_side=512,
        confirm_gap=0.01,
        veto_gap=0.03,
        minimum_support=0.55,
        rescue_margin=0.50,
        rescue_gap=0.00,
        rescue_budget=0.04,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prototype_bank_path = prototype_bank_path
        self.dino_extractor = DINOv3DenseExtractor(
            str(Path(dino_weights).resolve()),
            domain=dino_domain,
            device=str(self.device),
            max_side=dino_max_side,
        )
        self.confirm_gap = float(confirm_gap)
        self.veto_gap = float(veto_gap)
        self.minimum_support = float(minimum_support)
        self.rescue_margin = float(rescue_margin)
        self.rescue_gap = float(rescue_gap)
        self.rescue_budget = float(rescue_budget)
        self.prototype_bank = None
        self.prototype_valid = None
        if prototype_bank_path:
            self._load_prototype_bank(prototype_bank_path)
        self.probe = SAM3ProbeCapture(
            self.processor.model,
            top_queries=10,
        )
        self.num_decoder_layers = self.probe.num_layers

    def _schema_route_candidate(
        self,
        image,
        target_hw,
        official,
        presence_safe,
    ):
        if self.deployment_code == 0:
            return official.clone(), official.new_tensor(0.0).float()
        text_outputs, _ = self._inference_single_view_experts(image)
        text_outputs = {
            mode: self._resize_logits(logits, target_hw)
            for mode, logits in text_outputs.items()
        }
        text_official = self._prediction(text_outputs["official"]).long()
        parity = official.ne(text_official).float().amax()
        schema = self._prediction(
            text_outputs["schema_deploy"]
        ).long()
        if self.deployment_code == 2:
            schema = self._nonconflicting_union(
                official,
                schema,
                presence_safe,
            )
        return schema, parity

    @torch.inference_mode()
    def predict(self, inputs, data_samples):
        del inputs
        base_indices = {
            mode: index for index, mode in enumerate(HERA_BASE_MODES)
        }
        for sample in data_samples:
            meta = sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            target_hw = tuple(meta["ori_shape"][:2])
            identity_branches, identity_layers = self._view_with_layers(
                image
            )
            identity_branches, identity_layers = self._resize_view(
                identity_branches,
                identity_layers,
                target_hw,
            )
            rotated_branches, rotated_layers = self._view_with_layers(
                self._rotate_90(image)
            )
            rotated_branches = tuple(
                self._undo_rotate_90(output)
                for output in rotated_branches
            )
            rotated_layers = self._undo_rotate_90(rotated_layers)
            rotated_branches, _ = self._resize_view(
                rotated_branches,
                rotated_layers,
                target_hw,
            )

            _, semantic, instance, presence, _, _ = identity_branches
            logits, base_predictions, diagnostics, scale_summary = (
                self._build_multi_axis_predictions(
                    semantic,
                    instance,
                    presence,
                )
            )
            # Preserve the official crop-fusion/overlap-averaging order for P0.
            # Other response variants remain proposed candidates, not baselines.
            logits = self._aggregate_queries(identity_branches[0])
            base_predictions[0] = self._official_prediction(logits)
            rchf, rchf_errors = self._rchf_candidate(
                identity_branches,
                rotated_branches,
            )
            official = base_predictions[0].long()
            raw = {
                mode: (
                    rchf.long()
                    if mode == "rchf"
                    else base_predictions[base_indices[mode]].long()
                )
                for mode in HERA_EXPERT_MODES
            }
            schema_route, text_parity = self._schema_route_candidate(
                image,
                target_hw,
                official,
                raw["presence_safe"],
            )
            raw["schema_route"] = schema_route

            layer_logits = self._aggregate_layer_queries(identity_layers)
            layer_predictions = layer_logits.argmax(dim=1)
            layer_predictions[
                layer_logits.amax(dim=1) < self.prob_thd
            ] = self.bg_idx
            semantic_logits = self._aggregate_queries(semantic * presence)
            instance_logits = self._aggregate_queries(instance * presence)
            semantic_prediction = self._official_prediction(semantic_logits)
            instance_prediction = self._official_prediction(instance_logits)
            pool = torch.cat(
                [
                    layer_predictions[:-1].long(),
                    semantic_prediction.unsqueeze(0),
                    instance_prediction.unsqueeze(0),
                    torch.stack(tuple(raw.values())),
                ],
                dim=0,
            )

            features = self.dino_extractor.extract(image)["last"]
            similarity = self._prototype_similarity(features, target_hw)
            spatial_support = self._spatial_support(similarity)
            specs = {
                "balanced": (2, 0.01, 0.40, 0.06),
                "loose": (1, 0.00, 0.35, 0.08),
            }
            candidates = {}
            for mode, candidate in raw.items():
                _, veto = self._verify_candidate(
                    official,
                    candidate,
                    similarity,
                    spatial_support,
                )
                cfv = self._prototype_rescue(
                    logits,
                    official,
                    veto,
                    similarity,
                    spatial_support,
                )
                candidates[("cfv", mode)] = cfv
                for strength, spec in specs.items():
                    candidates[(f"trajectory_{strength}", mode)] = (
                        self._trajectory_candidate(
                            candidate,
                            pool,
                            similarity,
                            spatial_support,
                            *spec,
                        )
                    )
                    candidates[(f"cfv_trajectory_{strength}", mode)] = (
                        self._trajectory_candidate(
                            cfv,
                            pool,
                            similarity,
                            spatial_support,
                            *spec,
                        )
                    )
                    candidates[
                        (f"cfv_trajectory_add_{strength}", mode)
                    ] = self._trajectory_candidate(
                        cfv,
                        pool,
                        similarity,
                        spatial_support,
                        *spec,
                        eligible=cfv.eq(official),
                    )

            predictions = [official]
            for policy in POLICIES:
                for mode in EXPERT_MODES:
                    predictions.append(
                        raw[mode]
                        if policy == "raw"
                        else candidates[(policy, mode)]
                    )
            predictions = torch.stack(predictions).to(torch.int16)
            response_parity = official.ne(
                self._official_prediction(logits)
            ).float().amax()
            parity = torch.maximum(response_parity, text_parity)
            route_stats = torch.cat(
                [
                    scale_summary,
                    scale_summary[7].reshape(1),
                    rchf_errors,
                ]
            )
            sample.set_data(
                {
                    "seg_logits": PixelData(data=logits),
                    "pred_sem_seg": PixelData(
                        data=official.unsqueeze(0)
                    ),
                    "uctfs_candidate_preds": PixelData(
                        data=predictions
                    ),
                    "uctfs_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    "uctfs_route_stats": PixelData(
                        data=route_stats.reshape(1, -1)
                    ),
                    "multi_axis_diagnostics": PixelData(
                        data=diagnostics
                    ),
                }
            )
        return data_samples
