from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .PosterV2_7cls import (
    WindowAttentionGlobal,
    _to_channel_first,
    _to_channel_last,
    _to_query,
    feedforward,
    load_pretrain_file,
    load_pretrained_weights,
    window,
)
from .ir50 import Backbone
from .landmark_tokens import sample_keypoint_features
from .mobilefacenet import MobileFaceNet
from .vit_model import PatchEmbed


Tensor = torch.Tensor


class LandmarkGateFusion(nn.Module):
    """Lightweight landmark-driven gating over the three POSTER token groups."""

    def __init__(
        self,
        dims: Sequence[int],
        embed_dim: int,
        num_landmarks: int = 68,
        jitter_std: float = 0.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.jitter_std = jitter_std
        fused_input_dim = sum(dims) + 2
        hidden_dim = max(embed_dim // 4, 64)

        self.input_norm = nn.LayerNorm(fused_input_dim)
        self.reliability_head = nn.Sequential(
            nn.Linear(fused_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stage_context_proj = nn.ModuleList([nn.Linear(dim, embed_dim) for dim in dims])
        self.stage_gate_head = nn.ModuleList([nn.Linear(dim, 1) for dim in dims])
        self.context_norm = nn.LayerNorm(embed_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_scale = nn.Parameter(torch.zeros(1))
        self.context_scale = nn.Parameter(torch.zeros(1))

    def _prepare_landmarks(self, landmarks: Tensor) -> Tensor:
        landmarks = landmarks.view(landmarks.shape[0], self.num_landmarks, 2).clamp(0.0, 1.0)
        if self.training and self.jitter_std > 0:
            landmarks = (landmarks + torch.randn_like(landmarks) * self.jitter_std).clamp(0.0, 1.0)
        return landmarks

    def forward(
        self,
        token_groups: Sequence[Tensor],
        feature_maps: Sequence[Tensor],
        landmarks: Tensor,
    ) -> tuple[list[Tensor], Tensor, Tensor, Tensor]:
        normalized_landmarks = self._prepare_landmarks(landmarks)
        sampled_features = [sample_keypoint_features(feature_map, normalized_landmarks) for feature_map in feature_maps]
        fused_inputs = self.input_norm(torch.cat([*sampled_features, normalized_landmarks], dim=-1))
        reliability = torch.sigmoid(self.reliability_head(fused_inputs))
        normalization = reliability.sum(dim=1).clamp_min(1e-6)

        stage_summaries = [
            (stage_features * reliability).sum(dim=1) / normalization
            for stage_features in sampled_features
        ]
        stage_gate_logits = torch.cat(
            [gate_head(stage_summary) for gate_head, stage_summary in zip(self.stage_gate_head, stage_summaries)],
            dim=-1,
        )
        stage_gate = torch.softmax(stage_gate_logits, dim=-1)

        context = sum(
            stage_gate[:, index:index + 1] * stage_proj(stage_summary)
            for index, (stage_proj, stage_summary) in enumerate(zip(self.stage_context_proj, stage_summaries))
        )
        context = self.context_proj(self.context_norm(context))
        context = context.unsqueeze(1)

        gated_groups = []
        for index, group in enumerate(token_groups):
            scale = 1.0 + self.gate_scale * stage_gate[:, index].view(-1, 1, 1)
            residual = self.context_scale * stage_gate[:, index].view(-1, 1, 1) * context
            gated_groups.append(group * scale + residual)

        return gated_groups, reliability, normalized_landmarks, stage_gate


class KANLinear(nn.Module):
    """KAN-style linear layer with fixed triangular basis functions.

    Each output logit is an additive composition of per-feature univariate basis
    responses plus a small base linear path. This keeps the head lightweight
    while exposing feature-wise response curves through ``spline_weight``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_knots: int = 8,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        min_value, max_value = value_range
        if num_knots < 2:
            raise ValueError("num_knots must be at least 2.")
        grid = torch.linspace(min_value, max_value, num_knots)
        self.register_buffer("grid", grid)
        self.register_buffer("step", torch.tensor((max_value - min_value) / float(num_knots - 1)))

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_knots))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.trunc_normal_(self.spline_weight, std=0.02)
        nn.init.zeros_(self.base_bias)

    def _basis(self, x: Tensor) -> Tensor:
        step = float(self.step.item())
        x = torch.tanh(x)
        distances = torch.abs(x.unsqueeze(-1) - self.grid.view(1, 1, self.num_knots))
        return (1.0 - distances / max(step, 1e-6)).clamp_min(0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(x)
        base = F.linear(x, self.base_weight, self.base_bias)
        basis = self._basis(x)
        spline = torch.einsum("bik,oik->bo", basis, self.spline_weight)
        return base + spline


class KANHead(nn.Module):
    """Attention pooling plus a lightweight KAN classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_knots: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool_norm = nn.LayerNorm(input_dim)
        self.pool_score = nn.Linear(input_dim, 1)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.kan = KANLinear(hidden_dim, num_classes, num_knots=num_knots)
        self.last_pool_attention: Tensor | None = None
        self.last_pooled_features: Tensor | None = None
        self.last_projected_features: Tensor | None = None

        nn.init.trunc_normal_(self.pool_score.weight, std=0.02)
        nn.init.zeros_(self.pool_score.bias)
        nn.init.trunc_normal_(self.feature_proj[1].weight, std=0.02)
        nn.init.zeros_(self.feature_proj[1].bias)

    def forward(self, tokens: Tensor) -> Tensor:
        attention = torch.softmax(self.pool_score(self.pool_norm(tokens)), dim=1)
        pooled = (attention * tokens).sum(dim=1)
        self.last_pool_attention = attention.detach()
        self.last_pooled_features = pooled
        projected = self.feature_proj(pooled)
        self.last_projected_features = projected
        return self.kan(projected)


class POSTER_KAN(nn.Module):
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
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.window_size = list(window_size)
        self.num_heads = list(num_heads)
        self.dims = list(dims)
        self.dim_head = [dim // heads for dim, heads in zip(self.dims, self.num_heads)]
        self.N = [window_side * window_side for window_side in self.window_size]

        self.face_landback = MobileFaceNet([112, 112], 136, use_output_layer=True)
        face_landback_checkpoint = load_pretrain_file("mobilefacenet_model_best.pth.tar")
        self.face_landback = load_pretrained_weights(self.face_landback, face_landback_checkpoint)
        for parameter in self.face_landback.parameters():
            parameter.requires_grad = False

        self.ir_back = Backbone(50, 0.0, "ir", use_output_layer=False)
        ir_checkpoint = load_pretrain_file("ir50.pth")
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.attn1 = WindowAttentionGlobal(dim=dims[0], num_heads=num_heads[0], window_size=window_size[0])
        self.attn2 = WindowAttentionGlobal(dim=dims[1], num_heads=num_heads[1], window_size=window_size[1])
        self.attn3 = WindowAttentionGlobal(dim=dims[2], num_heads=num_heads[2], window_size=window_size[2])
        self.window1 = window(window_size=window_size[0], dim=dims[0])
        self.window2 = window(window_size=window_size[1], dim=dims[1])
        self.window3 = window(window_size=window_size[2], dim=dims[2])
        self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=3, stride=2, padding=1)

        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]
        self.ffn1 = feedforward(dim=dims[0], window_size=window_size[0], layer_scale=1e-5, drop_path=dpr[0])
        self.ffn2 = feedforward(dim=dims[1], window_size=window_size[1], layer_scale=1e-5, drop_path=dpr[1])
        self.ffn3 = feedforward(dim=dims[2], window_size=window_size[2], layer_scale=1e-5, drop_path=dpr[2])

        self.last_face_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.embed_q = nn.Sequential(
            nn.Conv2d(dims[0], embed_dim, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        )
        self.embed_k = nn.Sequential(nn.Conv2d(dims[1], embed_dim, kernel_size=3, stride=2, padding=1))
        self.embed_v = PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=embed_dim)

        self.landmark_gate_fusion = LandmarkGateFusion(
            dims=dims,
            embed_dim=embed_dim,
            num_landmarks=68,
            jitter_std=landmark_jitter,
        )
        self.kan_head = KANHead(
            input_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=kan_hidden_dim,
            num_knots=kan_num_knots,
        )

        self.last_landmark_reliability: Tensor | None = None
        self.last_landmarks: Tensor | None = None
        self.last_stage_gate: Tensor | None = None

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
        token_groups, landmark_reliability, normalized_landmarks, stage_gate = self.landmark_gate_fusion(
            [token_group1, token_group2, token_group3],
            landmark_feature_maps,
            face_landmarks,
        )

        self.last_landmark_reliability = landmark_reliability.detach()
        self.last_landmarks = normalized_landmarks.detach()
        self.last_stage_gate = stage_gate.detach()

        tokens = torch.cat(token_groups, dim=1)
        return self.kan_head(tokens)


__all__ = ["POSTER_KAN", "LandmarkGateFusion", "KANHead", "KANLinear"]
