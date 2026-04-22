from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .PosterV2_7cls import _to_channel_first, _to_channel_last, _to_query
from .poster_kan import KANLinear, POSTER_KAN


Tensor = torch.Tensor


class PrototypeAdaptiveKANHead(nn.Module):
    """Prototype-guided adaptive KAN classifier for expression tokens."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_knots: int = 8,
        num_prototypes: int = 7,
        prototype_scale_init: float = 0.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.pool_norm = nn.LayerNorm(input_dim)
        self.pool_score = nn.Linear(input_dim, 1)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.kan = KANLinear(hidden_dim, num_classes, num_knots=num_knots)
        self.expression_prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.prototype_gate = nn.Sequential(
            nn.LayerNorm(num_prototypes),
            nn.Linear(num_prototypes, max(num_prototypes * 2, 16)),
            nn.GELU(),
            nn.Linear(max(num_prototypes * 2, 16), num_classes),
            nn.Tanh(),
        )
        self.prototype_scale = nn.Parameter(torch.ones(1) * prototype_scale_init)
        self.last_pool_attention: Tensor | None = None
        self.last_pooled_features: Tensor | None = None
        self.last_projected_features: Tensor | None = None
        self.last_prototype_weights: Tensor | None = None

        nn.init.trunc_normal_(self.pool_score.weight, std=0.02)
        nn.init.zeros_(self.pool_score.bias)
        nn.init.trunc_normal_(self.feature_proj[1].weight, std=0.02)
        nn.init.zeros_(self.feature_proj[1].bias)
        nn.init.trunc_normal_(self.expression_prototypes, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        attention = torch.softmax(self.pool_score(self.pool_norm(tokens)), dim=1)
        pooled = (attention * tokens).sum(dim=1)
        self.last_pool_attention = attention.detach()
        self.last_pooled_features = pooled
        projected = self.feature_proj(pooled)
        self.last_projected_features = projected

        normalized_features = F.normalize(projected, dim=-1)
        normalized_prototypes = F.normalize(self.expression_prototypes, dim=-1)
        prototype_weights = torch.softmax(normalized_features @ normalized_prototypes.t(), dim=-1)
        class_gate = self.prototype_gate(prototype_weights)
        logits = self.kan(projected)
        self.last_prototype_weights = prototype_weights.detach()
        return logits * (1.0 + self.prototype_scale * class_gate)


class POSTER_KAN_PA(POSTER_KAN):
    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 7,
        window_size: Sequence[int] = (28, 14, 7),
        num_heads: Sequence[int] = (2, 4, 8),
        dims: Sequence[int] = (64, 128, 256),
        embed_dim: int = 768,
        landmark_jitter: float = 0.015,
        kan_hidden_dim: int = 256,
        kan_num_knots: int = 8,
        num_prototypes: int = 7,
        prototype_scale_init: float = 0.0,
        use_landmark_gate_fusion: bool = True,
    ) -> None:
        super().__init__(
            img_size=img_size,
            num_classes=num_classes,
            window_size=window_size,
            num_heads=num_heads,
            dims=dims,
            embed_dim=embed_dim,
            landmark_jitter=landmark_jitter,
            kan_hidden_dim=kan_hidden_dim,
            kan_num_knots=kan_num_knots,
        )
        self.kan_head = PrototypeAdaptiveKANHead(
            input_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=kan_hidden_dim,
            num_knots=kan_num_knots,
            num_prototypes=num_prototypes,
            prototype_scale_init=prototype_scale_init,
        )
        self.use_landmark_gate_fusion = use_landmark_gate_fusion

    def forward(self, x: Tensor) -> Tensor:
        x_face = F.interpolate(x, size=112)
        x_face1, x_face2, x_face3, face_landmarks = self.face_landback(x_face)
        x_face3 = self.last_face_conv(x_face3)
        x_face1 = _to_channel_last(x_face1)
        x_face2 = _to_channel_last(x_face2)
        x_face3 = _to_channel_last(x_face3)

        q1 = _to_query(x_face1, self.N[0], self.num_heads[0], self.dim_head[0])
        q2 = _to_query(x_face2, self.N[1], self.num_heads[1], self.dim_head[1])
        q3 = _to_query(x_face3, self.N[2], self.num_heads[2], self.dim_head[2])

        x_ir1, x_ir2, x_ir3 = self.ir_back(x)
        x_ir1, x_ir2, x_ir3 = self.conv1(x_ir1), self.conv2(x_ir2), self.conv3(x_ir3)
        x_window1, shortcut1 = self.window1(x_ir1)
        x_window2, shortcut2 = self.window2(x_ir2)
        x_window3, shortcut3 = self.window3(x_ir3)

        o1 = self.attn1(x_window1, q1)
        o2 = self.attn2(x_window2, q2)
        o3 = self.attn3(x_window3, q3)
        o1 = self.ffn1(o1, shortcut1)
        o2 = self.ffn2(o2, shortcut2)
        o3 = self.ffn3(o3, shortcut3)

        o1, o2, o3 = _to_channel_first(o1), _to_channel_first(o2), _to_channel_first(o3)
        landmark_feature_maps = [o1, o2, o3]

        token_group1 = self.embed_q(o1).flatten(2).transpose(1, 2)
        token_group2 = self.embed_k(o2).flatten(2).transpose(1, 2)
        token_group3 = self.embed_v(o3)
        token_groups = [token_group1, token_group2, token_group3]

        if self.use_landmark_gate_fusion:
            token_groups, landmark_reliability, normalized_landmarks, stage_gate = self.landmark_gate_fusion(
                token_groups,
                landmark_feature_maps,
                face_landmarks,
            )
        else:
            batch_size = face_landmarks.shape[0]
            normalized_landmarks = face_landmarks.view(batch_size, 68, 2).clamp(0.0, 1.0)
            landmark_reliability = torch.ones(batch_size, 68, 1, device=x.device, dtype=x.dtype)
            stage_gate = torch.full((batch_size, 3), 1.0 / 3.0, device=x.device, dtype=x.dtype)

        self.last_landmark_reliability = landmark_reliability.detach()
        self.last_landmarks = normalized_landmarks.detach()
        self.last_stage_gate = stage_gate.detach()

        tokens = torch.cat(token_groups, dim=1)
        return self.kan_head(tokens)


__all__ = ["POSTER_KAN_PA", "PrototypeAdaptiveKANHead"]
