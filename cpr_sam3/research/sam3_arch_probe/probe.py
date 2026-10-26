import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


ATTENTION_METRICS = ("entropy", "peak", "top1_mass")


class AttentionStatsRecorder:
    def __init__(self, num_layers: int, num_heads: int, top_queries: int = 10):
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.top_queries = int(top_queries)
        self.record_image = False
        self.record_prompt = False
        self._prompt_stats: Dict[int, torch.Tensor] = {}
        self._image_sum = torch.zeros(
            self.num_layers,
            self.num_heads,
            len(ATTENTION_METRICS),
            dtype=torch.float64,
        )
        self._image_count = torch.zeros(
            self.num_layers,
            self.num_heads,
            1,
            dtype=torch.float64,
        )

    def start_image(self, enabled: bool) -> None:
        self.record_image = bool(enabled)
        self.record_prompt = False
        self._prompt_stats.clear()
        self._image_sum.zero_()
        self._image_count.zero_()

    def start_prompt(self) -> None:
        self.record_prompt = self.record_image
        self._prompt_stats.clear()

    def observe(
        self,
        layer_idx: int,
        module: torch.nn.MultiheadAttention,
        args,
        kwargs,
    ) -> None:
        if not self.record_prompt:
            return

        query = kwargs.get("query", args[0] if len(args) > 0 else None)
        key = kwargs.get("key", args[1] if len(args) > 1 else None)
        attn_mask = kwargs.get("attn_mask")
        key_padding_mask = kwargs.get("key_padding_mask")
        if query is None or key is None:
            return

        with torch.no_grad():
            stats = self._compute_stats(
                module,
                query,
                key,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        self._prompt_stats[int(layer_idx)] = stats.cpu()

    def finish_prompt(self, layer_object_probs: torch.Tensor) -> None:
        if not self.record_prompt:
            return

        scores = layer_object_probs.detach().float().cpu()
        if scores.ndim == 3:
            scores = scores[:, 0]
        if scores.ndim != 2:
            raise ValueError(f"Expected [layers, queries] scores, got {scores.shape}.")

        for layer_idx, stats in self._prompt_stats.items():
            num_queries = scores.shape[-1]
            if stats.shape[1] == num_queries + 1:
                stats = stats[:, 1:]
            elif stats.shape[1] != num_queries:
                raise ValueError(
                    "Attention/query mismatch: "
                    f"attention={stats.shape[1]}, scores={num_queries}."
                )

            top_k = min(self.top_queries, num_queries)
            selected = scores[layer_idx].topk(top_k).indices
            selected_stats = stats[:, selected]
            self._image_sum[layer_idx] += selected_stats.sum(dim=1).double()
            self._image_count[layer_idx] += float(top_k)

        self.record_prompt = False
        self._prompt_stats.clear()

    def finish_image(self) -> Optional[torch.Tensor]:
        if not self.record_image:
            return None
        denom = self._image_count.clamp_min(1.0)
        stats = (self._image_sum / denom).float()
        stats[self._image_count.expand_as(stats) == 0] = float("nan")
        return stats

    @staticmethod
    def _compute_stats(
        module: torch.nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if module.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)

        if query.ndim != 3 or key.ndim != 3:
            raise ValueError("SAM3 decoder attention must use 3D query/key tensors.")
        if not getattr(module, "_qkv_same_embed_dim", True):
            raise NotImplementedError("Separate q/k projection dimensions are unsupported.")

        embed_dim = module.embed_dim
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads
        q_len, batch_size, _ = query.shape
        k_len = key.shape[0]

        weight = module.in_proj_weight
        bias = module.in_proj_bias
        q_bias = bias[:embed_dim] if bias is not None else None
        k_bias = bias[embed_dim : 2 * embed_dim] if bias is not None else None
        q_proj = F.linear(query, weight[:embed_dim], q_bias)
        k_proj = F.linear(key, weight[embed_dim : 2 * embed_dim], k_bias)

        q_proj = (
            q_proj.view(q_len, batch_size, num_heads, head_dim)
            .permute(1, 2, 0, 3)
            .float()
        )
        k_proj = (
            k_proj.view(k_len, batch_size, num_heads, head_dim)
            .permute(1, 2, 0, 3)
            .float()
        )
        logits = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        logits.mul_(head_dim**-0.5)

        if attn_mask is not None:
            mask = attn_mask
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask.reshape(batch_size, num_heads, q_len, k_len)
            else:
                raise ValueError(f"Unsupported attention mask shape {mask.shape}.")
            if mask.dtype == torch.bool:
                logits.masked_fill_(mask, float("-inf"))
            else:
                logits.add_(mask.float())

        if key_padding_mask is not None:
            padding = key_padding_mask
            if padding.shape == (k_len, batch_size):
                padding = padding.transpose(0, 1)
            padding = padding.to(torch.bool)[:, None, None, :]
            logits.masked_fill_(padding, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = entropy / math.log(max(k_len, 2))
        peak = probs.amax(dim=-1)
        top_count = max(1, math.ceil(k_len * 0.01))
        top1_mass = probs.topk(top_count, dim=-1).values.sum(dim=-1)

        stats = torch.stack((entropy, peak, top1_mass), dim=-1)
        return stats.mean(dim=0)


class SAM3ProbeCapture:
    def __init__(self, model, top_queries: int = 10):
        self.model = model
        decoder = model.transformer.decoder
        self.num_layers = len(decoder.layers)
        self.num_heads = decoder.layers[0].cross_attn.num_heads
        self.attention = AttentionStatsRecorder(
            self.num_layers,
            self.num_heads,
            top_queries=top_queries,
        )
        self.decoder_output = None
        self.raw_class_logits = None
        self.instance_embeds = None
        self._handles = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        self._handles.append(
            self.model.transformer.decoder.register_forward_hook(
                self._capture_decoder_output
            )
        )
        self._handles.append(
            self.model.dot_prod_scoring.register_forward_hook(
                self._capture_class_logits
            )
        )
        self._handles.append(
            self.model.segmentation_head.instance_seg_head.register_forward_hook(
                self._capture_instance_embeds
            )
        )
        for layer_idx, layer in enumerate(self.model.transformer.decoder.layers):
            self._handles.append(
                layer.cross_attn.register_forward_hook(
                    self._make_attention_hook(layer_idx),
                    with_kwargs=True,
                )
            )

    def reset_prompt(self) -> None:
        self.decoder_output = None
        self.raw_class_logits = None
        self.instance_embeds = None
        self.attention.start_prompt()

    def _capture_decoder_output(self, module, args, output) -> None:
        del module, args
        self.decoder_output = output

    def _capture_class_logits(self, module, args, output) -> None:
        del module, args
        self.raw_class_logits = output

    def _capture_instance_embeds(self, module, args, output) -> None:
        del module, args
        self.instance_embeds = output

    def _make_attention_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            del output
            self.attention.observe(layer_idx, module, args, kwargs)

        return hook

    def get_layer_tensors(self):
        if self.decoder_output is None:
            raise RuntimeError("Decoder output was not captured.")
        if self.raw_class_logits is None:
            raise RuntimeError("Class logits were not captured.")
        if self.instance_embeds is None:
            raise RuntimeError("Instance embeddings were not captured.")

        hs_seq, _, presence_logits, _ = self.decoder_output
        if presence_logits is None:
            raise RuntimeError("SAM3 presence logits are required for this probe.")

        hs = hs_seq.permute(0, 2, 1, 3)
        presence = presence_logits.permute(0, 2, 1).sigmoid().squeeze(-1)
        raw_probs = self.raw_class_logits.sigmoid().squeeze(-1)

        if self.model.supervise_joint_box_scores:
            joint_prob = raw_probs * presence.unsqueeze(-1)
            eps = 1e-3
            joint_prob = joint_prob.clamp(0.0, 1.0)
            joint_logits = torch.log(
                joint_prob.clamp_min(eps)
                / (1.0 - joint_prob).clamp_min(eps)
            ).clamp(min=-10.0, max=10.0)
            class_probs = joint_logits.sigmoid()
        else:
            class_probs = raw_probs

        object_probs = class_probs * presence.unsqueeze(-1)
        return hs, presence, object_probs, self.instance_embeds

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
