"""Expose a fixed deployable result, not a label-selected candidate."""

import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS

from cpr_sam3.research.cross_expert_trajectory_consensus.segmentor import (
    OUTPUT_MODES,
    CrossExpertTrajectorySegmentor,
)
from cpr_sam3.research.cross_foundation_candidate_verifier.segmentor import (
    CrossFoundationCandidateVerifierSegmentor,
)


@MODELS.register_module()
class CPRSAM3Segmentor(CrossExpertTrajectorySegmentor):
    _identity_outputs = CrossFoundationCandidateVerifierSegmentor._identity_outputs
    calibration_observation = CrossFoundationCandidateVerifierSegmentor.calibration_observation

    def __init__(self, *args, output_family, **kwargs):
        self.output_family = str(output_family)
        mode = f"evidence_max_{self.output_family}"
        if mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output: {mode}")
        self._result_index = OUTPUT_MODES.index(mode)
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        self.processor.model.eval().requires_grad_(False)
        self.dino_extractor.model.eval().requires_grad_(False)

    def _load_prototype_bank(self, path):
        from pathlib import Path
        import json

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "class_names" in payload:
            entries = json.loads(Path(self.prompt_bank_path).read_text(encoding="utf-8"))["classes"]
            if payload["class_names"] != [entry["name"] for entry in entries]:
                raise ValueError("Prototype bank class ordering does not match the selected prompt bank")
        super()._load_prototype_bank(path)

    @torch.inference_mode()
    def predict(self, inputs, data_samples):
        samples = super().predict(inputs, data_samples)
        for sample in samples:
            prediction = sample.cetc_candidate_preds.data[self._result_index].long()
            sample.set_data({"pred_sem_seg": PixelData(data=prediction.unsqueeze(0))})
            # The inherited logits score the baseline, not the reconciled output.
            if "seg_logits" in sample:
                del sample.seg_logits
        return samples
