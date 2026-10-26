from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.hierarchical_equivariance_response_sam3.segmentor import (
    HierarchicalEquivarianceResponseSegmentor,
)
from cpr_sam3.research.multi_axis_consensus_sam3.segmentor import (
    OUTPUT_MODES as HERA_BASE_MODES,
)
from cpr_sam3.research.satellite_affinity_sam3.dino import DINOv3DenseExtractor


EXPERT_MODES = (
    "presence_safe",
    "presence_vote",
    "direct_raw",
    "direct_safe",
    "direct_vote",
    "inverse_safe",
    "inverse_extend",
    "rchf",
)
POLICIES = ("raw", "confirm", "veto", "veto_rescue")
OUTPUT_MODES = ("official",) + tuple(
    f"{policy}_{mode}"
    for policy in POLICIES
    for mode in EXPERT_MODES
)


@MODELS.register_module()
class CrossFoundationCandidateVerifierSegmentor(
    HierarchicalEquivarianceResponseSegmentor
):
    """Use frozen DINOv3 prototypes to verify HERA corrections."""

    def __init__(
        self,
        *args,
        prototype_bank_path=None,
        dino_weights=(
            "checkpoints/"
            "dinov3_server_bundle/"
            "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        ),
        dino_domain="sat",
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

    def _load_prototype_bank(self, path):
        payload = torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=True,
        )
        prototypes = payload["prototypes"].float()
        valid = payload["valid"].bool()
        if prototypes.ndim == 2:
            prototypes = prototypes.unsqueeze(1)
            valid = valid.unsqueeze(1)
        if prototypes.shape[0] != self.num_cls:
            raise ValueError(
                "Prototype class count mismatch: "
                f"bank={prototypes.shape[0]}, model={self.num_cls}"
            )
        self.prototype_bank = F.normalize(
            prototypes.to(self.device), dim=-1, eps=1e-6
        )
        self.prototype_valid = valid.to(self.device)

    def _prototype_similarity(self, features, target_hw):
        if self.prototype_bank is None:
            raise RuntimeError(
                "prototype_bank_path is required for evaluation"
            )
        tokens = features.flatten(1).transpose(0, 1).float()
        similarity = torch.einsum(
            "ckd,nd->ckn", self.prototype_bank, tokens
        )
        similarity = similarity.masked_fill(
            ~self.prototype_valid.unsqueeze(-1), -1.0
        )
        similarity = similarity.max(dim=1).values.reshape(
            self.num_cls, *features.shape[-2:]
        )
        return F.interpolate(
            similarity.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @staticmethod
    def _spatial_support(similarity):
        choice = similarity.argmax(dim=0)
        one_hot = F.one_hot(
            choice, num_classes=similarity.shape[0]
        ).permute(2, 0, 1).float()
        return F.avg_pool2d(
            one_hot.unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2,
        ).squeeze(0)

    def _verify_candidate(
        self,
        official,
        candidate,
        similarity,
        support,
    ):
        candidate_similarity = similarity.gather(
            0, candidate.unsqueeze(0)
        ).squeeze(0)
        official_similarity = similarity.gather(
            0, official.unsqueeze(0)
        ).squeeze(0)
        candidate_support = support.gather(
            0, candidate.unsqueeze(0)
        ).squeeze(0)
        official_support = support.gather(
            0, official.unsqueeze(0)
        ).squeeze(0)
        gap = candidate_similarity - official_similarity
        changed = candidate.ne(official)

        confirmed = (
            changed
            & gap.gt(self.confirm_gap)
            & candidate_support.ge(self.minimum_support)
        )
        confirm = official.clone()
        confirm[confirmed] = candidate[confirmed]

        rejected = (
            changed
            & gap.lt(-self.veto_gap)
            & official_support.ge(self.minimum_support)
        )
        veto = candidate.clone()
        veto[rejected] = official[rejected]
        return confirm, veto

    def _prototype_rescue(
        self,
        logits,
        official,
        base,
        similarity,
        support,
    ):
        if logits.shape[0] < 2:
            return base.clone()
        top2 = logits.float().topk(2, dim=0)
        current = top2.indices[0]
        alternative = top2.indices[1]
        margin = (
            (top2.values[0] - top2.values[1])
            / top2.values[0].abs().clamp_min(0.05)
        )
        current_similarity = similarity.gather(
            0, current.unsqueeze(0)
        ).squeeze(0)
        alternative_similarity = similarity.gather(
            0, alternative.unsqueeze(0)
        ).squeeze(0)
        alternative_support = support.gather(
            0, alternative.unsqueeze(0)
        ).squeeze(0)
        gap = alternative_similarity - current_similarity
        selected = (
            base.eq(official)
            & official.eq(current)
            & margin.lt(self.rescue_margin)
            & gap.gt(self.rescue_gap)
            & similarity.argmax(dim=0).eq(alternative)
            & alternative_support.ge(self.minimum_support)
        )
        indices = torch.nonzero(
            selected.flatten(), as_tuple=False
        ).flatten()
        output = base.clone()
        if indices.numel() == 0:
            return output
        budget = max(
            1,
            int(round(output.numel() * self.rescue_budget)),
        )
        budget = min(budget, indices.numel())
        chosen = indices[gap.flatten()[indices].topk(budget).indices]
        flat_output = output.flatten()
        flat_alternative = alternative.flatten()
        flat_output[chosen] = flat_alternative[chosen]
        return output

    def _identity_outputs(self, image, target_hw):
        identity = self._resize_outputs(
            self._single_view_outputs(image), target_hw
        )
        logits, predictions, diagnostics, scale_summary = (
            self._build_multi_axis_predictions(
                identity[1], identity[2], identity[3]
            )
        )
        # OV3 fuses each crop before averaging overlapping windows. Rebuilding
        # the reference from averaged branches changes that order on OEM.
        logits = self._aggregate_queries(identity[0])
        predictions[0] = self._official_prediction(logits)
        return identity, logits, predictions, diagnostics, scale_summary

    @torch.inference_mode()
    def calibration_observation(self, image):
        target_hw = (image.height, image.width)
        _, logits, predictions, _, _ = self._identity_outputs(
            image, target_hw
        )
        features = self.dino_extractor.extract(image)["last"]
        return logits.float(), predictions[0].long(), features.float()

    def predict(self, inputs, data_samples):
        del inputs
        base_indices = {
            mode: index for index, mode in enumerate(HERA_BASE_MODES)
        }
        for sample in data_samples:
            meta = sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            target_hw = tuple(meta["ori_shape"][:2])
            (
                identity,
                logits,
                base_predictions,
                diagnostics,
                scale_summary,
            ) = self._identity_outputs(image, target_hw)
            rotated = self._single_view_outputs(self._rotate_90(image))
            aligned = self._resize_outputs(
                tuple(self._undo_rotate_90(output) for output in rotated),
                target_hw,
            )
            rchf_prediction, rchf_errors = self._rchf_candidate(
                identity, aligned
            )
            raw = {
                mode: (
                    rchf_prediction.long()
                    if mode == "rchf"
                    else base_predictions[base_indices[mode]].long()
                )
                for mode in EXPERT_MODES
            }
            official = base_predictions[0].long()
            features = self.dino_extractor.extract(image)["last"]
            similarity = self._prototype_similarity(features, target_hw)
            support = self._spatial_support(similarity)

            verified = {}
            for mode, candidate in raw.items():
                confirm, veto = self._verify_candidate(
                    official, candidate, similarity, support
                )
                verified[("confirm", mode)] = confirm
                verified[("veto", mode)] = veto
                verified[("veto_rescue", mode)] = self._prototype_rescue(
                    logits,
                    official,
                    veto,
                    similarity,
                    support,
                )

            predictions = [official]
            for policy in POLICIES:
                for mode in EXPERT_MODES:
                    predictions.append(
                        raw[mode]
                        if policy == "raw"
                        else verified[(policy, mode)]
                    )
            predictions = torch.stack(predictions).to(torch.int16)
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
                    "seg_logits": PixelData(data=logits),
                    "pred_sem_seg": PixelData(
                        data=official.unsqueeze(0)
                    ),
                    "cfv_candidate_preds": PixelData(data=predictions),
                    "cfv_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    "cfv_diagnostics": PixelData(data=diagnostics),
                    "cfv_route_stats": PixelData(
                        data=route_stats.reshape(1, -1)
                    ),
                }
            )
        return data_samples
