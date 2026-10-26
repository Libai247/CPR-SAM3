import json
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from segearthov3_segmentor import SegEarthOV3Segmentation


PROMPT_MODES = (
    "official",
    "expert_max",
    "expert_mean",
    "presence_top1",
    "presence_weighted",
    "positive_residual",
)


@MODELS.register_module()
class ConceptPromptExpertSegmentor(SegEarthOV3Segmentation):
    """Evaluate fixed remote-sensing prompt experts in one SAM3 pass."""

    output_modes = PROMPT_MODES

    def __init__(
        self,
        prompt_bank_path,
        presence_power=4.0,
        residual_alpha=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_bank_path = str(prompt_bank_path)
        self.presence_power = float(presence_power)
        self.residual_alpha = float(residual_alpha)
        (
            self.expert_words,
            self.expert_class_idx,
            self.official_prompt_mask,
            self.class_prompt_ranges,
        ) = self._load_prompt_bank(prompt_bank_path)

        self.expert_class_idx = torch.tensor(
            self.expert_class_idx,
            dtype=torch.long,
            device=self.device,
        )
        self.official_prompt_mask = torch.tensor(
            self.official_prompt_mask,
            dtype=torch.bool,
            device=self.device,
        )
        self._reported = False

    def _load_prompt_bank(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        classes = payload.get("classes")
        if not isinstance(classes, list) or len(classes) != self.num_cls:
            raise ValueError(
                f"{path} must define exactly {self.num_cls} class entries."
            )

        official_by_class = [[] for _ in range(self.num_cls)]
        for word, class_idx in zip(
            self.query_words,
            self.query_idx.detach().cpu().tolist(),
        ):
            official_by_class[class_idx].append(word)

        words = []
        class_indices = []
        official_mask = []
        class_ranges = []
        for class_idx, entry in enumerate(classes):
            prompts = entry.get("prompts", [])
            if not prompts or len(prompts) != len(set(prompts)):
                raise ValueError(
                    f"Class {class_idx} in {path} needs unique prompts."
                )
            missing = [
                prompt
                for prompt in official_by_class[class_idx]
                if prompt not in prompts
            ]
            if missing:
                raise ValueError(
                    f"Class {class_idx} is missing official prompts: {missing}"
                )

            start = len(words)
            for prompt in prompts:
                words.append(prompt)
                class_indices.append(class_idx)
                official_mask.append(
                    prompt in official_by_class[class_idx]
                )
            class_ranges.append((start, len(words)))

        return words, class_indices, official_mask, class_ranges

    @staticmethod
    def _resize_map(tensor, target_hw):
        if tensor.shape[-2:] == target_hw:
            return tensor
        return F.interpolate(
            tensor,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def _prompt_response(self, state, target_hw):
        height, width = target_hw
        fused = torch.zeros(
            (height, width),
            device=self.device,
            dtype=torch.float32,
        )

        if (
            self.use_transformer_decoder
            and state["masks_logits"].shape[0] > 0
        ):
            masks = self._resize_map(
                state["masks_logits"],
                target_hw,
            ).squeeze(1)
            scores = state["object_score"].reshape(-1, 1, 1)
            fused = torch.maximum(fused, (masks * scores).amax(dim=0).float())

        if self.use_sem_seg:
            semantic = self._resize_map(
                state["semantic_mask_logits"],
                target_hw,
            ).squeeze().float()
            fused = torch.maximum(fused, semantic)

        presence = torch.as_tensor(
            state["presence_score"],
            device=self.device,
        ).float().mean()
        if self.use_presence_score:
            fused = fused * presence
        return fused, presence

    def _infer_all_prompts(self, image):
        width, height = image.size
        target_hw = (height, width)
        logits = torch.zeros(
            (len(self.expert_words), height, width),
            device=self.device,
            dtype=torch.float32,
        )
        presence = torch.zeros(
            len(self.expert_words),
            device=self.device,
            dtype=torch.float32,
        )

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            state = self.processor.set_image(image)
            for prompt_idx, prompt in enumerate(self.expert_words):
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(
                    state=state,
                    prompt=prompt,
                )
                response, score = self._prompt_response(state, target_hw)
                logits[prompt_idx] = response
                presence[prompt_idx] = score
        return logits, presence

    def _aggregate_prompts(self, prompt_logits, presence):
        outputs = {
            mode: torch.zeros(
                (self.num_cls, *prompt_logits.shape[-2:]),
                device=self.device,
                dtype=prompt_logits.dtype,
            )
            for mode in self.output_modes
        }
        selected_prompts = torch.zeros(
            self.num_cls,
            device=self.device,
            dtype=torch.long,
        )

        for class_idx, (start, end) in enumerate(self.class_prompt_ranges):
            class_logits = prompt_logits[start:end]
            class_presence = presence[start:end].clamp_min(1e-6)
            class_official = self.official_prompt_mask[start:end]

            official = class_logits[class_official].amax(dim=0)
            expert_max = class_logits.amax(dim=0)
            expert_mean = class_logits.mean(dim=0)
            best_idx = int(class_presence.argmax().item())
            weights = class_presence.pow(self.presence_power)
            weights = weights / weights.sum().clamp_min(1e-6)
            weighted = (
                class_logits * weights.view(-1, 1, 1)
            ).sum(dim=0)
            positive_residual = (
                expert_max - official
            ).clamp_min(0.0)

            outputs["official"][class_idx] = official
            outputs["expert_max"][class_idx] = expert_max
            outputs["expert_mean"][class_idx] = expert_mean
            outputs["presence_top1"][class_idx] = class_logits[best_idx]
            outputs["presence_weighted"][class_idx] = weighted
            outputs["positive_residual"][class_idx] = (
                official + self.residual_alpha * positive_residual
            )
            selected_prompts[class_idx] = start + best_idx

        return outputs, selected_prompts

    def _inference_single_view_experts(self, image):
        prompt_logits, presence = self._infer_all_prompts(image)
        return self._aggregate_prompts(prompt_logits, presence)

    def _slide_inference_experts(self, image):
        width, height = image.size
        stride = (
            (self.slide_stride, self.slide_stride)
            if isinstance(self.slide_stride, int)
            else self.slide_stride
        )
        crop = (
            (self.slide_crop, self.slide_crop)
            if isinstance(self.slide_crop, int)
            else self.slide_crop
        )
        h_stride, w_stride = stride
        h_crop, w_crop = crop
        accumulators = {
            mode: torch.zeros(
                (self.num_cls, height, width),
                device=self.device,
                dtype=torch.float32,
            )
            for mode in self.output_modes
        }
        counts = torch.zeros(
            (1, height, width),
            device=self.device,
            dtype=torch.float32,
        )
        selected_counts = torch.zeros(
            (self.num_cls, len(self.expert_words)),
            device=self.device,
            dtype=torch.long,
        )
        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, height)
                x2 = min(x1 + w_crop, width)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_outputs, selected = self._inference_single_view_experts(
                    image.crop((x1, y1, x2, y2))
                )
                for mode in self.output_modes:
                    accumulators[mode][:, y1:y2, x1:x2] += (
                        crop_outputs[mode]
                    )
                counts[:, y1:y2, x1:x2] += 1
                selected_counts[
                    torch.arange(self.num_cls, device=self.device),
                    selected,
                ] += 1

        if (counts == 0).any():
            raise RuntimeError("Sparse sliding-window coverage.")
        outputs = {
            mode: values / counts
            for mode, values in accumulators.items()
        }
        selected = selected_counts.argmax(dim=1)
        return outputs, selected

    def _prediction(self, logits):
        prediction = logits.argmax(dim=0)
        prediction[logits.amax(dim=0) < self.prob_thd] = self.bg_idx
        return prediction

    def predict(self, inputs, data_samples):
        del inputs
        for sample in data_samples:
            image = Image.open(sample.metainfo["img_path"]).convert("RGB")
            ori_shape = sample.metainfo["ori_shape"]
            use_slide = (
                self.slide_crop > 0
                and (
                    self.slide_crop < image.size[0]
                    or self.slide_crop < image.size[1]
                )
            )
            if use_slide:
                outputs, selected = self._slide_inference_experts(image)
            else:
                outputs, selected = self._inference_single_view_experts(image)

            for mode in self.output_modes:
                if outputs[mode].shape[-2:] != ori_shape:
                    outputs[mode] = F.interpolate(
                        outputs[mode].unsqueeze(0),
                        size=ori_shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

            predictions = torch.stack(
                [
                    self._prediction(outputs[mode])
                    for mode in self.output_modes
                ],
                dim=0,
            )
            official_logits = outputs["official"]
            official_pred = predictions[
                self.output_modes.index("official")
            ]
            sample.set_data(
                {
                    "seg_logits": PixelData(data=official_logits),
                    "pred_sem_seg": PixelData(
                        data=official_pred.unsqueeze(0)
                    ),
                    "prompt_expert_preds": PixelData(data=predictions),
                }
            )

        if not self._reported:
            print(
                "[ConceptPromptExperts] "
                f"bank={self.prompt_bank_path} "
                f"prompts={len(self.expert_words)} "
                f"modes={','.join(self.output_modes)}"
            )
            self._reported = True
        return data_samples
