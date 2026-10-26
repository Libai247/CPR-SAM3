from collections import OrderedDict

import torch
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.p3c_sam3.segmentor import P3CSegmentor


MODE_SPECS = OrderedDict(
    [
        ("official", None),
        ("p3c_safe", None),
        ("p3c_wide", None),
        ("g075_official", None),
        ("g075_p3c_safe", None),
        ("g075_p3c_wide", None),
        ("g050_official", None),
        ("g050_p3c_safe", None),
        ("g050_p3c_wide", None),
        ("agree_safe", None),
        ("agree_wide", None),
        ("mix10_safe", None),
        ("mix10_wide", None),
        ("mix25_safe", None),
        ("mix25_wide", None),
        ("mix50_safe", None),
        ("mix50_wide", None),
        ("union_safe", None),
        ("union_wide", None),
        ("product_safe", None),
        ("product_wide", None),
        ("instance_safe", None),
        ("instance_wide", None),
    ]
)
OUTPUT_MODES = tuple(MODE_SPECS)


@MODELS.register_module()
class BranchP3CSegmentor(P3CSegmentor):
    """Top-2 cross-class calibration using separate SAM3 branches."""

    def _query_outputs(self, state, target_hw):
        height, width = target_hw
        semantic = torch.zeros((height, width), device=self.device)
        instance = torch.zeros_like(semantic)

        if self.use_sem_seg:
            semantic = self._resize_map(
                state["semantic_mask_logits"],
                target_hw,
            ).squeeze()

        if self.use_transformer_decoder and state["masks_logits"].shape[0] > 0:
            masks = self._resize_map(
                state["masks_logits"],
                target_hw,
            ).squeeze(1)
            scores = state["object_score"].reshape(-1, 1, 1)
            instance = (masks * scores).amax(dim=0)

        presence = torch.as_tensor(
            state["presence_score"],
            device=self.device,
            dtype=semantic.dtype,
        )
        official = torch.maximum(semantic, instance)
        if self.use_presence_score:
            official = official * presence
        raw = torch.maximum(semantic, instance)
        return (
            official,
            semantic,
            instance,
            presence.expand_as(semantic),
            raw * presence.pow(0.75),
            raw * presence.pow(0.50),
        )

    def _inference_single_view_branches(self, image):
        width, height = image.size
        shape = (self.num_queries, height, width)
        outputs = [
            torch.zeros(shape, device=self.device)
            for _ in range(6)
        ]

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            state = self.processor.set_image(image)
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(
                    state=state,
                    prompt=query_word,
                )
                query_outputs = self._query_outputs(
                    state,
                    (height, width),
                )
                for target, source in zip(outputs, query_outputs):
                    target[query_idx] = source
        return tuple(outputs)

    def _slide_inference_branches(self, image):
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
        shape = (self.num_queries, height, width)
        outputs = [
            torch.zeros(shape, device=self.device)
            for _ in range(6)
        ]
        counts = torch.zeros((1, height, width), device=self.device)
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
                crop_outputs = self._inference_single_view_branches(
                    image.crop((x1, y1, x2, y2))
                )
                for target, source in zip(outputs, crop_outputs):
                    target[:, y1:y2, x1:x2] += source
                counts[:, y1:y2, x1:x2] += 1

        if (counts == 0).any():
            raise RuntimeError("Sparse sliding-window coverage.")
        return tuple(output / counts for output in outputs)

    def _calibrate(self, official, official_pred, evidence, wide=False):
        return self._calibrated_prediction(
            official,
            official_pred,
            evidence,
            weight=0.50 if wide else 0.75,
            margin=0.50 if wide else 0.20,
            top2_only=True,
        )

    def _top2_context(self, official, evidence):
        top2 = official.float().topk(
            min(2, official.shape[0]),
            dim=0,
        ).indices
        if top2.shape[0] == 1:
            top2 = top2.expand(2, -1, -1)
        pair = evidence.gather(0, top2)
        choice = top2.gather(
            0,
            pair.argmax(dim=0, keepdim=True),
        ).squeeze(0)
        confidence = (
            (pair.amax(dim=0) - pair.amin(dim=0))
            / pair.amax(dim=0).clamp_min(self.calibration_eps)
        )
        return top2, choice, confidence

    def _agreement_prediction(
        self,
        official,
        official_pred,
        semantic_evidence,
        instance_evidence,
        margin,
    ):
        values, top2 = official.float().topk(
            min(2, official.shape[0]),
            dim=0,
        )
        if top2.shape[0] < 2:
            return official_pred.clone()
        sem_pair = semantic_evidence.gather(0, top2)
        inst_pair = instance_evidence.gather(0, top2)
        sem_index = sem_pair.argmax(dim=0, keepdim=True)
        inst_index = inst_pair.argmax(dim=0, keepdim=True)
        candidate = top2.gather(0, sem_index).squeeze(0)
        relative_margin = (
            (values[0] - values[1])
            / torch.maximum(values[0].abs(), values[1].abs()).clamp_min(
                self.calibration_eps
            )
        )
        sem_gap = (
            (sem_pair.amax(dim=0) - sem_pair.amin(dim=0))
            / sem_pair.amax(dim=0).clamp_min(self.calibration_eps)
        )
        inst_gap = (
            (inst_pair.amax(dim=0) - inst_pair.amin(dim=0))
            / inst_pair.amax(dim=0).clamp_min(self.calibration_eps)
        )
        selected = (
            sem_index.eq(inst_index).squeeze(0)
            & candidate.ne(official_pred)
            & (relative_margin <= float(margin))
            & (sem_gap >= 0.05)
            & (inst_gap >= 0.10)
            & (inst_pair.amax(dim=0) >= 0.10)
        )
        output = official_pred.clone()
        output[selected] = candidate[selected]
        return output

    def _build_predictions(
        self,
        official,
        semantic,
        instance,
        presence,
        gamma075,
        gamma050,
    ):
        official_pred = self._official_prediction(official)
        gamma075_pred = self._official_prediction(gamma075)
        gamma050_pred = self._official_prediction(gamma050)
        semantic_evidence = self._robust_evidence(
            semantic,
            presence,
            0.99,
            use_presence=True,
        )
        semantic_evidence_g075 = self._robust_evidence(
            semantic,
            presence.float().pow(0.75),
            0.99,
            use_presence=True,
        )
        semantic_evidence_g050 = self._robust_evidence(
            semantic,
            presence.float().pow(0.50),
            0.99,
            use_presence=True,
        )
        instance_evidence = self._robust_evidence(
            instance,
            presence,
            0.99,
            use_presence=True,
        )

        mix10 = 0.90 * semantic_evidence + 0.10 * instance_evidence
        mix25 = 0.75 * semantic_evidence + 0.25 * instance_evidence
        mix50 = 0.50 * semantic_evidence + 0.50 * instance_evidence
        union = torch.maximum(semantic_evidence, instance_evidence)
        product = torch.sqrt(
            semantic_evidence.clamp_min(self.calibration_eps)
            * instance_evidence.clamp_min(self.calibration_eps)
        )

        predictions = [
            official_pred,
            self._calibrate(
                official,
                official_pred,
                semantic_evidence,
            ),
            self._calibrate(
                official,
                official_pred,
                semantic_evidence,
                wide=True,
            ),
            gamma075_pred,
            self._calibrate(
                gamma075,
                gamma075_pred,
                semantic_evidence_g075,
            ),
            self._calibrate(
                gamma075,
                gamma075_pred,
                semantic_evidence_g075,
                wide=True,
            ),
            gamma050_pred,
            self._calibrate(
                gamma050,
                gamma050_pred,
                semantic_evidence_g050,
            ),
            self._calibrate(
                gamma050,
                gamma050_pred,
                semantic_evidence_g050,
                wide=True,
            ),
            self._agreement_prediction(
                official,
                official_pred,
                semantic_evidence,
                instance_evidence,
                margin=0.20,
            ),
            self._agreement_prediction(
                official,
                official_pred,
                semantic_evidence,
                instance_evidence,
                margin=0.50,
            ),
        ]
        for evidence in (
            mix10,
            mix25,
            mix50,
            union,
            product,
            instance_evidence,
        ):
            predictions.extend(
                [
                    self._calibrate(
                        official,
                        official_pred,
                        evidence,
                    ),
                    self._calibrate(
                        official,
                        official_pred,
                        evidence,
                        wide=True,
                    ),
                ]
            )

        top2, semantic_choice, _ = self._top2_context(
            official,
            semantic_evidence,
        )
        _, instance_choice, _ = self._top2_context(
            official,
            instance_evidence,
        )
        _, fusion_choice, _ = self._top2_context(
            official,
            mix25,
        )
        diagnostics = torch.stack(
            [
                top2[0],
                top2[1],
                semantic_choice,
                instance_choice,
                fusion_choice,
            ]
        ).to(torch.int16)
        return torch.stack(predictions).to(torch.int16), diagnostics

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
            predictions, diagnostics = self._build_predictions(*aggregated)
            official = aggregated[0]
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
                    "branch_p3c_preds": PixelData(data=predictions),
                    "branch_p3c_diagnostics": PixelData(data=diagnostics),
                    "branch_p3c_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                }
            )
        return data_samples
