from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from segearthov3_segmentor import SegEarthOV3Segmentation


MODE_SPECS = OrderedDict(
    [
        ("official", None),
        (
            "raw_sem_m020",
            dict(anchor="raw", weight=0.50, margin=0.20),
        ),
        (
            "q95_w025_m020",
            dict(anchor=0.95, weight=0.25, margin=0.20),
        ),
        (
            "q95_w050_m020",
            dict(anchor=0.95, weight=0.50, margin=0.20),
        ),
        (
            "q95_w075_m020",
            dict(anchor=0.95, weight=0.75, margin=0.20),
        ),
        (
            "q99_w025_m020",
            dict(anchor=0.99, weight=0.25, margin=0.20),
        ),
        (
            "q99_w050_m020",
            dict(anchor=0.99, weight=0.50, margin=0.20),
        ),
        (
            "q99_w075_m020",
            dict(anchor=0.99, weight=0.75, margin=0.20),
        ),
        (
            "q95_w050_m050",
            dict(anchor=0.95, weight=0.50, margin=0.50),
        ),
        (
            "q99_w050_m050",
            dict(anchor=0.99, weight=0.50, margin=0.50),
        ),
        (
            "q95_no_presence",
            dict(
                anchor=0.95,
                weight=0.50,
                margin=0.50,
                use_presence=False,
            ),
        ),
        (
            "q95_all_class",
            dict(
                anchor=0.95,
                weight=0.50,
                margin=0.50,
                top2_only=False,
            ),
        ),
    ]
)
OUTPUT_MODES = tuple(MODE_SPECS)


@MODELS.register_module()
class P3CSegmentor(SegEarthOV3Segmentation):
    """Presence-preserved cross-class competition for frozen SAM3."""

    def __init__(self, calibration_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.calibration_eps = float(calibration_eps)

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

    def _query_outputs(self, state, target_hw):
        height, width = target_hw
        semantic = torch.zeros(
            (height, width),
            device=self.device,
        )
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
        presence_map = presence.expand_as(semantic)
        return official, semantic, presence_map

    def _inference_single_view_p3c(self, image):
        width, height = image.size
        shape = (self.num_queries, height, width)
        official = torch.zeros(shape, device=self.device)
        semantic = torch.zeros_like(official)
        presence = torch.zeros_like(official)

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
                outputs = self._query_outputs(
                    state,
                    (height, width),
                )
                official[query_idx], semantic[query_idx], presence[query_idx] = (
                    outputs
                )
        return official, semantic, presence

    def _slide_inference_p3c(self, image):
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
        outputs = [torch.zeros(shape, device=self.device) for _ in range(3)]
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
                crop_outputs = self._inference_single_view_p3c(
                    image.crop((x1, y1, x2, y2))
                )
                for target, source in zip(outputs, crop_outputs):
                    target[:, y1:y2, x1:x2] += source
                counts[:, y1:y2, x1:x2] += 1

        if (counts == 0).any():
            raise RuntimeError("Sparse sliding-window coverage.")
        return tuple(output / counts for output in outputs)

    def _aggregate_queries(self, tensor):
        output = torch.zeros(
            (self.num_cls, *tensor.shape[-2:]),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        for class_idx in range(self.num_cls):
            output[class_idx] = tensor[self.query_idx == class_idx].amax(dim=0)
        return output

    def _official_prediction(self, logits):
        prediction = logits.argmax(dim=0)
        prediction[logits.amax(dim=0) < self.prob_thd] = self.bg_idx
        return prediction

    def _robust_evidence(
        self,
        semantic,
        presence,
        quantile,
        use_presence=True,
    ):
        semantic = semantic.float()
        class_count = semantic.shape[0]
        flat = semantic.flatten(1)
        q50 = torch.quantile(flat, 0.50, dim=1).view(class_count, 1, 1)
        upper = torch.quantile(
            flat,
            float(quantile),
            dim=1,
        ).view(class_count, 1, 1)
        evidence = (
            (semantic - q50)
            / (upper - q50).clamp_min(self.calibration_eps)
        ).clamp_(0.0, 1.0)
        if use_presence:
            evidence = evidence * presence.float().clamp_(0.0, 1.0)
        return evidence

    def _calibrated_prediction(
        self,
        official,
        official_pred,
        evidence,
        weight,
        margin,
        top2_only=True,
    ):
        official = official.float().clamp_min(self.calibration_eps)
        top_values, top_indices = official.topk(
            min(2, official.shape[0]),
            dim=0,
        )
        if top_values.shape[0] < 2:
            return official_pred.clone()

        relative_margin = (
            (top_values[0] - top_values[1])
            / torch.maximum(
                top_values[0].abs(),
                top_values[1].abs(),
            ).clamp_min(self.calibration_eps)
        )
        selected = (
            (relative_margin <= float(margin))
            & (top_values[0] >= self.prob_thd)
        )

        if top2_only:
            candidate_evidence = evidence.gather(0, top_indices)
            candidate_scores = (
                top_values.clamp_min(self.calibration_eps).log()
                + float(weight)
                * candidate_evidence.clamp_min(self.calibration_eps).log()
            )
            selected_index = candidate_scores.argmax(dim=0, keepdim=True)
            calibrated = top_indices.gather(0, selected_index).squeeze(0)
        else:
            scores = (
                official.log()
                + float(weight)
                * evidence.clamp_min(self.calibration_eps).log()
            )
            calibrated = scores.argmax(dim=0)

        prediction = official_pred.clone()
        prediction[selected] = calibrated[selected]
        return prediction

    def _build_predictions(self, official, semantic, presence):
        official_pred = self._official_prediction(official)
        evidence_cache = {}
        predictions = []

        for mode, spec in MODE_SPECS.items():
            if mode == "official":
                predictions.append(official_pred)
                continue

            anchor = spec["anchor"]
            use_presence = spec.get("use_presence", True)
            cache_key = (anchor, use_presence)
            if cache_key not in evidence_cache:
                if anchor == "raw":
                    evidence = semantic.float().clamp_min(0.0)
                    if use_presence:
                        evidence = evidence * presence.float().clamp(0.0, 1.0)
                    evidence_cache[cache_key] = evidence
                else:
                    evidence_cache[cache_key] = self._robust_evidence(
                        semantic,
                        presence,
                        anchor,
                        use_presence=use_presence,
                    )

            predictions.append(
                self._calibrated_prediction(
                    official,
                    official_pred,
                    evidence_cache[cache_key],
                    weight=spec["weight"],
                    margin=spec["margin"],
                    top2_only=spec.get("top2_only", True),
                )
            )

        return torch.stack(predictions).to(torch.int16)

    def predict(self, inputs, data_samples):
        del inputs
        for data_sample in data_samples:
            meta = data_sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            if self.slide_crop > 0 and (
                self.slide_crop < image.size[0]
                or self.slide_crop < image.size[1]
            ):
                query_outputs = self._slide_inference_p3c(image)
            else:
                query_outputs = self._inference_single_view_p3c(image)

            ori_shape = tuple(meta["ori_shape"][:2])
            if query_outputs[0].shape[-2:] != ori_shape:
                query_outputs = tuple(
                    self._resize_map(output.unsqueeze(0), ori_shape).squeeze(0)
                    for output in query_outputs
                )

            official, semantic, presence = (
                self._aggregate_queries(output) for output in query_outputs
            )
            predictions = self._build_predictions(
                official,
                semantic,
                presence,
            )
            official_pred = predictions[0].long()
            parity = official_pred.ne(
                self._official_prediction(official)
            ).float().amax()

            data_sample.set_data(
                {
                    "seg_logits": PixelData(data=official),
                    "pred_sem_seg": PixelData(data=official_pred.unsqueeze(0)),
                    "p3c_preds": PixelData(data=predictions),
                    "p3c_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                }
            )

        return data_samples
