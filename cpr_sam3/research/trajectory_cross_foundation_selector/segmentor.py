import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from cpr_sam3.research.cross_foundation_candidate_verifier.segmentor import (
    EXPERT_MODES,
    HERA_BASE_MODES,
    CrossFoundationCandidateVerifierSegmentor,
)
from cpr_sam3.research.sam3_arch_probe.probe import SAM3ProbeCapture


POLICIES = (
    "raw",
    "cfv",
    "trajectory_strict",
    "trajectory_balanced",
    "trajectory_loose",
    "cfv_trajectory_strict",
    "cfv_trajectory_balanced",
    "cfv_trajectory_add_strict",
    "cfv_trajectory_add_balanced",
)
OUTPUT_MODES = ("official",) + tuple(
    f"{policy}_{mode}"
    for policy in POLICIES
    for mode in EXPERT_MODES
)


@MODELS.register_module()
class TrajectoryCrossFoundationSegmentor(
    CrossFoundationCandidateVerifierSegmentor
):
    """Select SAM3 trajectory candidates with frozen DINOv3 evidence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe = SAM3ProbeCapture(
            self.processor.model,
            top_queries=10,
        )
        self.num_decoder_layers = self.probe.num_layers

    def _layer_query_logits(self, state, target_hw):
        hs, presence, object_probs, instance_embeds = (
            self.probe.get_layer_tensors()
        )
        semantic = self._resize_map(
            state["semantic_mask_logits"], target_hw
        ).squeeze()
        layers = torch.zeros(
            (self.num_decoder_layers, *target_hw),
            device=self.device,
            dtype=semantic.dtype,
        )
        for layer_index in range(self.num_decoder_layers):
            fused = semantic.clone() if self.use_sem_seg else torch.zeros_like(semantic)
            if self.use_transformer_decoder:
                scores = object_probs[layer_index, 0]
                keep = scores > self.confidence_threshold
                if keep.any():
                    mask_logits = (
                        self.processor.model.segmentation_head.mask_predictor(
                            hs[layer_index], instance_embeds
                        )[:, keep]
                    )
                    mask_logits = self._resize_map(mask_logits, target_hw)
                    masks = mask_logits.sigmoid().squeeze(0)
                    instance = (
                        masks * scores[keep].reshape(-1, 1, 1)
                    ).amax(dim=0)
                    fused = torch.maximum(fused, instance)
            if self.use_presence_score:
                fused = fused * presence[layer_index, 0]
            layers[layer_index] = fused
        return layers

    def _inference_view_with_layers(self, image):
        width, height = image.size
        target_hw = (height, width)
        shape = (self.num_queries, height, width)
        branches = [torch.zeros(shape, device=self.device) for _ in range(6)]
        layers = torch.zeros(
            (
                self.num_decoder_layers,
                self.num_queries,
                height,
                width,
            ),
            device=self.device,
        )
        self.probe.attention.start_image(False)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            state = self.processor.set_image(image)
            for query_index, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(state)
                self.probe.reset_prompt()
                state = self.processor.set_text_prompt(
                    state=state, prompt=query_word
                )
                for target, source in zip(
                    branches, self._query_outputs(state, target_hw)
                ):
                    target[query_index] = source
                layers[:, query_index] = self._layer_query_logits(
                    state, target_hw
                )
        self.probe.attention.finish_image()
        return tuple(branches), layers

    def _slide_view_with_layers(self, image):
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
        branches = [torch.zeros(shape, device=self.device) for _ in range(6)]
        layers = torch.zeros(
            (
                self.num_decoder_layers,
                self.num_queries,
                height,
                width,
            ),
            device=self.device,
        )
        counts = torch.zeros((1, height, width), device=self.device)
        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_index in range(h_grids):
            for w_index in range(w_grids):
                y1 = h_index * h_stride
                x1 = w_index * w_stride
                y2 = min(y1 + h_crop, height)
                x2 = min(x1 + w_crop, width)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_branches, crop_layers = self._inference_view_with_layers(
                    image.crop((x1, y1, x2, y2))
                )
                for target, source in zip(branches, crop_branches):
                    target[:, y1:y2, x1:x2] += source
                layers[:, :, y1:y2, x1:x2] += crop_layers
                counts[:, y1:y2, x1:x2] += 1
        if (counts == 0).any():
            raise RuntimeError("Sparse sliding-window coverage.")
        return (
            tuple(output / counts for output in branches),
            layers / counts.unsqueeze(0),
        )

    def _view_with_layers(self, image):
        if self.slide_crop > 0 and (
            self.slide_crop < image.size[0]
            or self.slide_crop < image.size[1]
        ):
            return self._slide_view_with_layers(image)
        return self._inference_view_with_layers(image)

    def _resize_view(self, branches, layers, target_hw):
        if branches[0].shape[-2:] == target_hw:
            return branches, layers
        branches = tuple(
            self._resize_map(output.unsqueeze(0), target_hw).squeeze(0)
            for output in branches
        )
        flat = layers.flatten(0, 1).unsqueeze(1)
        flat = self._resize_map(flat, target_hw).squeeze(1)
        layers = flat.view(
            self.num_decoder_layers,
            self.num_queries,
            *target_hw,
        )
        return branches, layers

    def _trajectory_candidate(
        self,
        base,
        pool,
        similarity,
        spatial_support,
        minimum_votes,
        similarity_gap,
        minimum_local_support,
        change_budget,
        eligible=None,
    ):
        votes = torch.zeros(
            (self.num_cls, *base.shape),
            device=base.device,
            dtype=torch.int16,
        )
        ones = torch.ones_like(base.unsqueeze(0), dtype=torch.int16)
        for prediction in pool:
            votes.scatter_add_(0, prediction.long().unsqueeze(0), ones)
        score = similarity.float() + 0.025 * votes.float()
        score = score.masked_fill(votes.eq(0), -2.0)
        alternative = score.argmax(dim=0)
        support = votes.gather(
            0, alternative.unsqueeze(0)
        ).squeeze(0)
        base_similarity = similarity.gather(
            0, base.unsqueeze(0)
        ).squeeze(0)
        alternative_similarity = similarity.gather(
            0, alternative.unsqueeze(0)
        ).squeeze(0)
        local_support = spatial_support.gather(
            0, alternative.unsqueeze(0)
        ).squeeze(0)
        gap = alternative_similarity - base_similarity
        selected = (
            alternative.ne(base)
            & support.ge(int(minimum_votes))
            & gap.gt(float(similarity_gap))
            & local_support.ge(float(minimum_local_support))
        )
        if eligible is not None:
            selected &= eligible
        indices = torch.nonzero(selected.flatten(), as_tuple=False).flatten()
        output = base.clone()
        if indices.numel() == 0:
            return output
        budget = max(1, int(round(output.numel() * float(change_budget))))
        budget = min(budget, indices.numel())
        priority = gap + support.float() * 0.01
        chosen = indices[priority.flatten()[indices].topk(budget).indices]
        flat_output = output.flatten()
        flat_alternative = alternative.flatten()
        flat_output[chosen] = flat_alternative[chosen]
        return output

    def _aggregate_layer_queries(self, layer_queries):
        output = torch.zeros(
            (
                layer_queries.shape[0],
                self.num_cls,
                *layer_queries.shape[-2:],
            ),
            device=layer_queries.device,
            dtype=layer_queries.dtype,
        )
        for class_index in range(self.num_cls):
            selected = self.query_idx == class_index
            output[:, class_index] = layer_queries[:, selected].amax(dim=1)
        return output

    def predict(self, inputs, data_samples):
        del inputs
        base_indices = {
            mode: index for index, mode in enumerate(HERA_BASE_MODES)
        }
        for sample in data_samples:
            meta = sample.metainfo
            image = Image.open(meta["img_path"]).convert("RGB")
            target_hw = tuple(meta["ori_shape"][:2])
            identity_branches, identity_layers = self._view_with_layers(image)
            identity_branches, identity_layers = self._resize_view(
                identity_branches, identity_layers, target_hw
            )
            rotated_branches, rotated_layers = self._view_with_layers(
                self._rotate_90(image)
            )
            rotated_branches = tuple(
                self._undo_rotate_90(output) for output in rotated_branches
            )
            rotated_layers = self._undo_rotate_90(rotated_layers)
            rotated_branches, _ = self._resize_view(
                rotated_branches, rotated_layers, target_hw
            )

            _, semantic, instance, presence, _, _ = identity_branches
            logits, base_predictions, diagnostics, scale_summary = (
                self._build_multi_axis_predictions(
                    semantic, instance, presence
                )
            )
            rchf, rchf_errors = self._rchf_candidate(
                identity_branches, rotated_branches
            )
            official = base_predictions[0].long()
            raw = {
                mode: (
                    rchf.long()
                    if mode == "rchf"
                    else base_predictions[base_indices[mode]].long()
                )
                for mode in EXPERT_MODES
            }

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
            candidates = {}
            for mode, candidate in raw.items():
                _, veto = self._verify_candidate(
                    official, candidate, similarity, spatial_support
                )
                cfv = self._prototype_rescue(
                    logits,
                    official,
                    veto,
                    similarity,
                    spatial_support,
                )
                candidates[("cfv", mode)] = cfv
                specs = {
                    "trajectory_strict": (3, 0.02, 0.45, 0.04),
                    "trajectory_balanced": (2, 0.01, 0.40, 0.06),
                    "trajectory_loose": (1, 0.00, 0.35, 0.08),
                }
                for policy, spec in specs.items():
                    candidates[(policy, mode)] = self._trajectory_candidate(
                        candidate,
                        pool,
                        similarity,
                        spatial_support,
                        *spec,
                    )
                for policy, spec in (
                    ("cfv_trajectory_strict", specs["trajectory_strict"]),
                    ("cfv_trajectory_balanced", specs["trajectory_balanced"]),
                ):
                    candidates[(policy, mode)] = self._trajectory_candidate(
                        cfv,
                        pool,
                        similarity,
                        spatial_support,
                        *spec,
                    )
                for policy, spec in (
                    (
                        "cfv_trajectory_add_strict",
                        specs["trajectory_strict"],
                    ),
                    (
                        "cfv_trajectory_add_balanced",
                        specs["trajectory_balanced"],
                    ),
                ):
                    candidates[(policy, mode)] = self._trajectory_candidate(
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
            sample.set_data(
                {
                    "seg_logits": PixelData(data=logits),
                    "pred_sem_seg": PixelData(data=official.unsqueeze(0)),
                    "tcfs_candidate_preds": PixelData(data=predictions),
                    "tcfs_parity": PixelData(
                        data=parity.reshape(1, 1, 1)
                    ),
                    "tcfs_diagnostics": PixelData(data=diagnostics),
                    "tcfs_route_stats": PixelData(
                        data=route_stats.reshape(1, -1)
                    ),
                }
            )
        return data_samples
