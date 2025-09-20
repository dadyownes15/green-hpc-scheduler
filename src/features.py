from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


@dataclass
class AttentionPoolConfig:
    """Thin wrapper for attention pooling hyperparameters."""

    input_dim: int
    hidden_dim: int
    dropout: float = 0.0


class AttentionPool(nn.Module):
    """One-layer attention pooling block with optional masking."""

    def __init__(self, config: AttentionPoolConfig) -> None:
        super().__init__()
        layers = [nn.Linear(config.input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.ReLU()]
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        self.feature_proj = nn.Sequential(*layers)
        self.score = nn.Linear(config.hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inputs: (batch, seq, input_dim)
        hidden = self.feature_proj(inputs)
        scores = self.score(hidden).squeeze(-1)

        if mask is not None:
            valid_mask = mask.bool()
            scores = scores.masked_fill(~valid_mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            weights = weights * valid_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            weights = torch.softmax(scores, dim=-1)

        pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        return pooled


class AttentionPoolFeaturesExtractor(BaseFeaturesExtractor):
    """Custom features extractor that pools structured segments with attention."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        config: Dict[str, Any],
        queue_hidden_dim: int = 128,
        running_hidden_dim: int = 64,
        forecast_hidden_dim: int = 64,
        carbon_context_dim: int = 32,
        final_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        assert isinstance(observation_space, gym.spaces.Box), "AttentionPoolFeaturesExtractor supports Box spaces only."
        assert config is not None, "Config dict must be provided to AttentionPoolFeaturesExtractor."

        self.max_queue_size = int(config["max_queue_size"])
        self.job_feature = int(config["job_feature"])
        self.run_win_length = int(config["run_win_length"])
        self.run_feature = int(config["run_feature"])
        self.green_constant_features = int(config["green_feature_constant"])
        self.green_forecast_length = int(config["green_forecast_length"])
        self.green_features_per_slot = int(config["green_feature_pr_timeslot"])

        self.queue_dim = self.max_queue_size * self.job_feature
        self.running_dim = self.run_win_length * self.run_feature
        self.forecast_steps = max(self.green_forecast_length - 1, 0)
        self.forecast_dim = self.forecast_steps * self.green_features_per_slot

        expected_dim = self.queue_dim + self.running_dim + self.green_constant_features + self.forecast_dim
        assert observation_space.shape is not None
        assert observation_space.shape[0] == expected_dim, (
            f"Observation shape mismatch: expected {expected_dim}, got {observation_space.shape[0]}"
        )

        super().__init__(observation_space, features_dim=final_dim)

        self.queue_pool = AttentionPool(
            AttentionPoolConfig(input_dim=self.job_feature, hidden_dim=queue_hidden_dim, dropout=dropout)
        )
        self.running_pool = AttentionPool(
            AttentionPoolConfig(input_dim=self.run_feature, hidden_dim=running_hidden_dim, dropout=dropout)
        )
        self.forecast_pool = AttentionPool(
            AttentionPoolConfig(input_dim=max(self.green_features_per_slot, 1), hidden_dim=forecast_hidden_dim, dropout=dropout)
        )

        carbon_layers = [nn.Linear(self.green_constant_features, carbon_context_dim), nn.LayerNorm(carbon_context_dim), nn.ReLU()]
        if dropout > 0:
            carbon_layers.append(nn.Dropout(dropout))
        self.carbon_context = nn.Sequential(*carbon_layers)

        combined_dim = queue_hidden_dim + running_hidden_dim + forecast_hidden_dim + carbon_context_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, final_dim),
            nn.ReLU(),
            nn.LayerNorm(final_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)

        queue_end = self.queue_dim
        running_end = queue_end + self.running_dim
        carbon_end = running_end + self.green_constant_features

        queue_flat = observations[:, :queue_end]
        running_flat = observations[:, queue_end:running_end]
        carbon_constant = observations[:, running_end:carbon_end]
        forecast_flat = observations[:, carbon_end:carbon_end + self.forecast_dim]

        queue = queue_flat.view(batch_size, self.max_queue_size, self.job_feature)
        running = running_flat.view(batch_size, self.run_win_length, self.run_feature)

        queue_mask = queue.abs().sum(dim=-1) > 0
        running_mask = running.abs().sum(dim=-1) > 0

        queue_repr = self.queue_pool(queue, queue_mask)
        running_repr = self.running_pool(running, running_mask)

        if self.forecast_dim > 0:
            forecast_seq = forecast_flat.view(batch_size, self.forecast_steps, max(self.green_features_per_slot, 1))
            forecast_repr = self.forecast_pool(forecast_seq)
        else:
            forecast_repr = torch.zeros(batch_size, 0, device=observations.device)

        carbon_repr = self.carbon_context(carbon_constant)

        features = torch.cat([queue_repr, running_repr, forecast_repr, carbon_repr], dim=-1)
        return self.final_mlp(features)
