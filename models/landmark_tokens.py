import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_keypoint_features(feature_map, normalized_landmarks):
    grid = normalized_landmarks.mul(2.0).sub(1.0).unsqueeze(2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2)


class ReliabilityWeightedKeypointTokens(nn.Module):
    def __init__(self, dims, embed_dim, num_landmarks=68, dropout=0.1, jitter_std=0.0):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.jitter_std = jitter_std
        self.input_dim = sum(dims) + 2

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        hidden_dim = max(embed_dim // 4, 64)
        self.reliability_head = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _prepare_landmarks(self, landmarks):
        landmarks = landmarks.view(landmarks.shape[0], self.num_landmarks, 2).clamp(0.0, 1.0)
        if self.training and self.jitter_std > 0:
            noise = torch.randn_like(landmarks) * self.jitter_std
            landmarks = (landmarks + noise).clamp(0.0, 1.0)
        return landmarks

    def forward(self, feature_maps, landmarks):
        landmarks = self._prepare_landmarks(landmarks)
        sampled_features = [sample_keypoint_features(feature_map, landmarks) for feature_map in feature_maps]
        fused_features = torch.cat([*sampled_features, landmarks], dim=-1)
        fused_features = self.input_norm(fused_features)

        reliability = torch.sigmoid(self.reliability_head(fused_features))
        landmark_tokens = self.token_proj(fused_features) * reliability
        return landmark_tokens, reliability, landmarks


class KeypointGuidedTokenFusion(nn.Module):
    def __init__(
        self,
        dims,
        embed_dim,
        num_landmarks=68,
        num_heads=8,
        dropout=0.1,
        jitter_std=0.0,
    ):
        super().__init__()
        self.keypoint_tokens = ReliabilityWeightedKeypointTokens(
            dims=dims,
            embed_dim=embed_dim,
            num_landmarks=num_landmarks,
            dropout=dropout,
            jitter_std=jitter_std,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, feature_maps, landmarks):
        landmark_tokens, reliability, landmarks = self.keypoint_tokens(feature_maps, landmarks)
        attn_out, _ = self.cross_attn(
            self.query_norm(tokens),
            self.key_norm(landmark_tokens),
            self.key_norm(landmark_tokens),
            need_weights=False,
        )
        fused_tokens = tokens + self.gamma * self.out_proj(attn_out)
        return fused_tokens, reliability, landmarks
