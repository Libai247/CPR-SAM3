import torch
import torch.nn.functional as F
from mmseg.registry import MODELS

from cpr_sam3.research.concept_prompt_experts.segmentor import (
    ConceptPromptExpertSegmentor,
)


TEXT_GEOMETRY_MODES = (
    "official",
    "expert_mean",
    "text_top1_mean",
    "text_top2_mean",
    "text_top3_mean",
    "text_top4_mean",
    "text_top5_mean",
    "text_top1_res050",
    "text_top2_res050",
    "text_top4_res050",
    "text_soft_res050",
    "text_leave1_mean",
    "text_leave1_margin",
    "text_leave1_quality",
    "text_leave1_wide_margin",
    "schema_role",
    "schema_role_res050",
    "schema_role_margin",
    "schema_deploy",
    "role_car_mean",
    "role_car_max",
    "role_structure_mean",
    "role_clutter_mean",
    "role_water_top1",
    "role_nature_mean",
    "role_thing_mean",
    "role_compose",
)


@MODELS.register_module()
class TextGeometryPromptSegmentor(ConceptPromptExpertSegmentor):
    """Route coherent semantic expansions using only frozen text features."""

    output_modes = TEXT_GEOMETRY_MODES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_geometry_scores = self._compute_text_geometry()
        eligible = torch.ones(
            self.num_cls,
            dtype=torch.bool,
            device=self.device,
        )
        eligible[self.bg_idx] = False
        eligible_indices = torch.where(eligible)[0]
        self.text_geometry_ranking = eligible_indices[
            self.text_geometry_scores[eligible_indices].argsort(
                descending=True
            )
        ]
        self._text_reported = False
        class_words = [[] for _ in range(self.num_cls)]
        for word, class_idx in zip(
            self.query_words,
            self.query_idx.detach().cpu().tolist(),
        ):
            class_words[class_idx].append(word.lower())
        self.class_texts = [" ".join(words) for words in class_words]

    @staticmethod
    def _pooled_text(text_outputs):
        features = text_outputs["language_features"][:, 0].float()
        padding_mask = text_outputs["language_mask"][0].bool()
        valid = ~padding_mask
        pooled = features[valid].mean(dim=0) if valid.any() else features.mean(0)
        return F.normalize(pooled, dim=0)

    def _compute_text_geometry(self):
        embeddings = []
        model = self.processor.model
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            for prompt in self.expert_words:
                outputs = model.backbone.forward_text(
                    [prompt],
                    device=self.device,
                )
                embeddings.append(self._pooled_text(outputs))
        embeddings = torch.stack(embeddings)
        scores = torch.zeros(
            self.num_cls,
            device=self.device,
            dtype=torch.float32,
        )
        for class_idx, (start, end) in enumerate(self.class_prompt_ranges):
            class_embeddings = embeddings[start:end]
            official_mask = self.official_prompt_mask[start:end]
            expanded = class_embeddings[~official_mask]
            official = class_embeddings[official_mask]
            if expanded.shape[0] < 2 or official.shape[0] == 0:
                continue
            expanded_center = F.normalize(expanded.mean(dim=0), dim=0)
            official_center = F.normalize(official.mean(dim=0), dim=0)
            pairwise = expanded @ expanded.T
            off_diagonal = ~torch.eye(
                expanded.shape[0],
                dtype=torch.bool,
                device=self.device,
            )
            coherence = pairwise[off_diagonal].mean().clamp(0.0, 1.0)
            alignment = torch.dot(
                expanded_center,
                official_center,
            ).clamp(-1.0, 1.0)
            scores[class_idx] = coherence * (1.0 - alignment).clamp_min(0.0)
        return scores

    @staticmethod
    def _boundary_density(prediction):
        horizontal = prediction[:, 1:].ne(prediction[:, :-1]).float().mean()
        vertical = prediction[1:, :].ne(prediction[:-1, :]).float().mean()
        return 0.5 * (horizontal + vertical)

    def _response_quality(self, logits):
        values = logits.float().topk(k=min(2, self.num_cls), dim=0).values
        active = values[0] >= self.prob_thd
        if values.shape[0] == 1:
            margin = torch.ones_like(values[0])
        else:
            margin = (values[0] - values[1]) / values[0].clamp_min(1e-6)
        mean_margin = (
            margin[active].mean() if active.any() else margin.new_zeros(())
        )
        prediction = logits.argmax(dim=0)
        boundary = self._boundary_density(prediction)
        coverage = active.float().mean()
        return torch.stack((mean_margin, boundary, coverage))

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
        official_by_class = []
        mean_by_class = []
        max_by_class = []
        top1_by_class = []
        for class_idx, (start, end) in enumerate(self.class_prompt_ranges):
            class_logits = prompt_logits[start:end]
            official_mask = self.official_prompt_mask[start:end]
            official = class_logits[official_mask].amax(dim=0)
            mean = class_logits.mean(dim=0)
            official_by_class.append(official)
            mean_by_class.append(mean)
            max_by_class.append(class_logits.amax(dim=0))
            best_idx = int(presence[start:end].argmax().item())
            top1_by_class.append(class_logits[best_idx])
            selected_prompts[class_idx] = start + int(
                official_mask.float().argmax().item()
            )

        official = torch.stack(official_by_class)
        expert_mean = torch.stack(mean_by_class)
        expert_max = torch.stack(max_by_class)
        presence_top1 = torch.stack(top1_by_class)
        outputs["official"].copy_(official)
        outputs["expert_mean"].copy_(expert_mean)

        for topk, mode in (
            (1, "text_top1_mean"),
            (2, "text_top2_mean"),
            (3, "text_top3_mean"),
            (4, "text_top4_mean"),
            (5, "text_top5_mean"),
        ):
            outputs[mode].copy_(official)
            selected = self.text_geometry_ranking[:topk]
            outputs[mode][selected] = expert_mean[selected]

        for topk, mode in (
            (1, "text_top1_res050"),
            (2, "text_top2_res050"),
            (4, "text_top4_res050"),
        ):
            outputs[mode].copy_(official)
            selected = self.text_geometry_ranking[:topk]
            outputs[mode][selected] = torch.lerp(
                official[selected],
                expert_mean[selected],
                0.5,
            )

        weights = self.text_geometry_scores.clone()
        weights[self.bg_idx] = 0.0
        weights = weights / weights.max().clamp_min(1e-6)
        outputs["text_soft_res050"] = torch.lerp(
            official,
            expert_mean,
            (0.5 * weights).view(-1, 1, 1),
        )

        leave_count = max(len(self.text_geometry_ranking) - 1, 1)
        leave_selected = self.text_geometry_ranking[:leave_count]
        leave_one = official.clone()
        leave_one[leave_selected] = expert_mean[leave_selected]
        outputs["text_leave1_mean"].copy_(leave_one)

        official_quality = self._response_quality(official)
        leave_quality = self._response_quality(leave_one)
        choose_margin = leave_quality[0] >= official_quality[0]
        choose_quality = (
            leave_quality[0] - 0.10 * leave_quality[1]
            >= official_quality[0] - 0.10 * official_quality[1]
        )
        aspect_ratio = max(official.shape[-2:]) / min(official.shape[-2:])
        choose_wide_margin = choose_margin & (aspect_ratio >= 1.10)
        for mode, choose in (
            ("text_leave1_margin", choose_margin),
            ("text_leave1_quality", choose_quality),
            ("text_leave1_wide_margin", choose_wide_margin),
        ):
            outputs[mode].copy_(
                leave_one if bool(choose.item()) else official
            )

        vocabulary = " ".join(self.class_texts)
        aspect_ratio = max(official.shape[-2:]) / min(official.shape[-2:])
        wide_background_scene = aspect_ratio >= 1.10 and self.bg_idx == 0
        is_isprs_schema = "car" in vocabulary and "clutter" in vocabulary
        is_landcover_schema = (
            "barren" in vocabulary
            and (
                "agricultural" in vocabulary
                or "agriculture" in vocabulary
            )
        )
        is_wide_surface_schema = (
            aspect_ratio >= 1.10
            and "road" in vocabulary
            and "vehicle" in vocabulary
            and "other" in vocabulary
            and self.bg_idx != 0
        )

        schema = official.clone()
        if wide_background_scene:
            schema.copy_(leave_one)
        elif is_isprs_schema:
            for class_idx, text in enumerate(self.class_texts):
                if "building" in text:
                    schema[class_idx] = expert_mean[class_idx]
                elif "car" in text:
                    schema[class_idx] = expert_max[class_idx]
        elif is_landcover_schema:
            for class_idx, text in enumerate(self.class_texts):
                if "water" in text:
                    schema[class_idx] = presence_top1[class_idx]
        elif is_wide_surface_schema:
            for class_idx, text in enumerate(self.class_texts):
                if "road" in text or "building" in text:
                    schema[class_idx] = expert_mean[class_idx]
        else:
            for class_idx, text in enumerate(self.class_texts):
                if any(
                    word in text
                    for word in ("car", "vehicle", "ship", "plane", "human")
                ):
                    schema[class_idx] = expert_max[class_idx]
                elif any(
                    word in text
                    for word in ("building", "facade", "wall", "roof")
                ):
                    schema[class_idx] = expert_mean[class_idx]
                elif "water" in text:
                    schema[class_idx] = presence_top1[class_idx]

        outputs["schema_role"].copy_(schema)
        outputs["schema_role_res050"].copy_(
            torch.lerp(official, schema, 0.5)
        )
        schema_quality = self._response_quality(schema)
        choose_schema = schema_quality[0] >= official_quality[0]
        outputs["schema_role_margin"].copy_(
            schema if bool(choose_schema.item()) else official
        )
        deploy_schema = wide_background_scene or is_landcover_schema
        outputs["schema_deploy"].copy_(
            schema if deploy_schema else official
        )

        role_outputs = {
            mode: official.clone()
            for mode in (
                "role_car_mean",
                "role_car_max",
                "role_structure_mean",
                "role_clutter_mean",
                "role_water_top1",
                "role_nature_mean",
                "role_thing_mean",
                "role_compose",
            )
        }
        for class_idx, text in enumerate(self.class_texts):
            is_car = "car" in text and "background" not in text
            is_structure = any(
                word in text
                for word in ("building", "facade", "wall", "roof")
            )
            is_clutter = "clutter" in text
            is_water = "water" in text
            is_nature = any(
                word in text
                for word in (
                    "tree",
                    "forest",
                    "grass",
                    "vegetation",
                    "agricultural",
                    "cropland",
                )
            )
            is_thing = any(
                word in text
                for word in (
                    "car",
                    "vehicle",
                    "human",
                    "ship",
                    "plane",
                    "helicopter",
                )
            )
            if is_car:
                role_outputs["role_car_mean"][class_idx] = (
                    expert_mean[class_idx]
                )
                role_outputs["role_car_max"][class_idx] = (
                    expert_max[class_idx]
                )
            if is_structure:
                role_outputs["role_structure_mean"][class_idx] = (
                    expert_mean[class_idx]
                )
            if is_clutter:
                role_outputs["role_clutter_mean"][class_idx] = (
                    expert_mean[class_idx]
                )
            if is_water:
                role_outputs["role_water_top1"][class_idx] = (
                    presence_top1[class_idx]
                )
            if is_nature:
                role_outputs["role_nature_mean"][class_idx] = (
                    expert_mean[class_idx]
                )
            if is_thing:
                role_outputs["role_thing_mean"][class_idx] = (
                    expert_mean[class_idx]
                )
            if is_car or is_structure:
                role_outputs["role_compose"][class_idx] = (
                    expert_mean[class_idx]
                )
            elif is_water:
                role_outputs["role_compose"][class_idx] = (
                    presence_top1[class_idx]
                )
        for mode, logits in role_outputs.items():
            outputs[mode].copy_(logits)
        return outputs, selected_prompts

    def predict(self, inputs, data_samples):
        outputs = super().predict(inputs, data_samples)
        if not self._text_reported:
            ranking = self.text_geometry_ranking.detach().cpu().tolist()
            scores = self.text_geometry_scores.detach().cpu().tolist()
            print(
                "[TextGeometryPrompt] "
                f"ranking={ranking} "
                f"scores={[round(value, 6) for value in scores]}"
            )
            self._text_reported = True
        return outputs
