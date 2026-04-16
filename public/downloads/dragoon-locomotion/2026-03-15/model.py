from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class RunningNorm(nn.Module):
    """
    간단한 running mean/std normalization.
    학습 중에는 update 가능, 추론 시에는 고정 사용.
    """
    def __init__(self, shape, eps: float = 1e-5, clip: float = 10.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.eps = eps
        self.clip = clip

        self.register_buffer("count", torch.tensor(eps))
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return

        dims = tuple(range(x.ndim - len(self.mean.shape)))
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)
        batch_count = torch.tensor(
            x.numel() / self.mean.numel(),
            device=x.device,
            dtype=self.mean.dtype,
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=self.eps))
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(x, -self.clip, self.clip)

    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update and self.training:
            self.update(x)
        return self.normalize(x)


class TerrainEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, activation: str = "elu"):
        super().__init__()
        act = get_activation(activation)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            act,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            act,
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 512),
            act,
            nn.LayerNorm(512),
            nn.Linear(512, out_dim),
            act,
        )

    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        x = self.conv(terrain)
        x = self.fc(x)
        return x


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int = 66, out_dim: int = 128, activation: str = "elu"):
        super().__init__()
        act = get_activation(activation)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            act,
            nn.Linear(128, 128),
            act,
            nn.LayerNorm(128),
            nn.Linear(128, out_dim),
            act,
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int, activation: str = "elu"):
        super().__init__()
        act = get_activation(activation)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DragoonActorCriticCfg:
    proprio_dim: int = 66
    terrain_channels: int = 1
    terrain_h: int = 21
    terrain_w: int = 21
    action_dim: int = 16

    proprio_feat_dim: int = 128
    terrain_feat_dim: int = 256

    fusion_hidden_dims: Tuple[int, ...] = (512, 512, 256)
    actor_hidden_dims: Tuple[int, ...] = (256, 128)
    critic_hidden_dims: Tuple[int, ...] = (256, 128)

    activation: str = "elu"
    init_log_std: float = -0.7
    min_log_std: float = -5.0
    max_log_std: float = 2.0

    use_obs_norm: bool = True


class DragoonActorCritic(nn.Module):
    def __init__(self, cfg: DragoonActorCriticCfg):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim

        self.use_obs_norm = cfg.use_obs_norm
        if self.use_obs_norm:
            self.proprio_norm = RunningNorm((cfg.proprio_dim,))
            self.terrain_norm = RunningNorm((cfg.terrain_channels, cfg.terrain_h, cfg.terrain_w))
        else:
            self.proprio_norm = None
            self.terrain_norm = None

        self.proprio_encoder = ProprioEncoder(
            input_dim=cfg.proprio_dim,
            out_dim=cfg.proprio_feat_dim,
            activation=cfg.activation,
        )
        self.terrain_encoder = TerrainEncoder(
            out_dim=cfg.terrain_feat_dim,
            activation=cfg.activation,
        )

        fusion_dim = cfg.proprio_feat_dim + cfg.terrain_feat_dim

        self.actor_fusion = MLPHead(
            input_dim=fusion_dim,
            hidden_dims=cfg.fusion_hidden_dims,
            output_dim=cfg.fusion_hidden_dims[-1],
            activation=cfg.activation,
        )

        self.critic_fusion = MLPHead(
            input_dim=fusion_dim,
            hidden_dims=cfg.fusion_hidden_dims,
            output_dim=cfg.fusion_hidden_dims[-1],
            activation=cfg.activation,
        )

        actor_last_dim = cfg.fusion_hidden_dims[-1]
        critic_last_dim = cfg.fusion_hidden_dims[-1]

        if len(cfg.actor_hidden_dims) > 0:
            self.actor_head_body = MLPHead(
                input_dim=actor_last_dim,
                hidden_dims=cfg.actor_hidden_dims,
                output_dim=cfg.actor_hidden_dims[-1],
                activation=cfg.activation,
            )
            actor_last_dim = cfg.actor_hidden_dims[-1]
        else:
            self.actor_head_body = nn.Identity()

        if len(cfg.critic_hidden_dims) > 0:
            self.critic_head_body = MLPHead(
                input_dim=critic_last_dim,
                hidden_dims=cfg.critic_hidden_dims,
                output_dim=cfg.critic_hidden_dims[-1],
                activation=cfg.activation,
            )
            critic_last_dim = cfg.critic_hidden_dims[-1]
        else:
            self.critic_head_body = nn.Identity()

        self.actor_mean = nn.Linear(actor_last_dim, cfg.action_dim)
        self.critic_value = nn.Linear(critic_last_dim, 1)

        self.log_std = nn.Parameter(torch.ones(cfg.action_dim) * cfg.init_log_std)
        
        self.min_log_std = cfg.min_log_std
        self.max_log_std = cfg.max_log_std

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)
        nn.init.constant_(self.critic_value.bias, 0.0)

    def _normalize_obs(self, obs: Dict[str, torch.Tensor], update_stats: bool) -> Dict[str, torch.Tensor]:
        proprio = obs["proprio"]
        terrain = obs["terrain"]

        if self.use_obs_norm:
            proprio = self.proprio_norm(proprio, update=update_stats)
            terrain = self.terrain_norm(terrain, update=update_stats)

        return {
            "proprio": proprio,
            "terrain": terrain,
        }

    def encode(self, obs: Dict[str, torch.Tensor], update_obs_stats: bool = False):
        obs = self._normalize_obs(obs, update_stats=update_obs_stats)

        proprio = obs["proprio"]
        terrain = obs["terrain"]

        proprio_feat = self.proprio_encoder(proprio)
        terrain_feat = self.terrain_encoder(terrain)
        fused = torch.cat([proprio_feat, terrain_feat], dim=-1)

        return proprio_feat, terrain_feat, fused

    def actor_forward(self, obs: Dict[str, torch.Tensor], update_obs_stats: bool = False):
        _, _, fused = self.encode(obs, update_obs_stats=update_obs_stats)

        x = self.actor_fusion(fused)
        x = self.actor_head_body(x)

        mean = self.actor_mean(x)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)

        return mean, log_std

    def critic_forward(self, obs: Dict[str, torch.Tensor], update_obs_stats: bool = False):
        _, _, fused = self.encode(obs, update_obs_stats=update_obs_stats)

        x = self.critic_fusion(fused)
        x = self.critic_head_body(x)

        value = self.critic_value(x)
        return value

    def forward(self, obs: Dict[str, torch.Tensor], update_obs_stats: bool = False):
        mean, log_std = self.actor_forward(obs, update_obs_stats=update_obs_stats)
        value = self.critic_forward(obs, update_obs_stats=update_obs_stats)
        return mean, log_std, value

    def get_dist(self, obs: Dict[str, torch.Tensor], update_obs_stats: bool = False) -> Normal:
        mean, log_std = self.actor_forward(obs, update_obs_stats=update_obs_stats)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    @torch.no_grad()
    def act_inference(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = self.actor_forward(obs, update_obs_stats=False)
        return torch.tanh(mean)

    def act(self, obs: Dict[str, torch.Tensor]):
        mean, log_std = self.actor_forward(obs, update_obs_stats=True)
        value = self.critic_forward(obs, update_obs_stats=False)

        std = torch.exp(log_std).expand_as(mean)
        dist = Normal(mean, std)

        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)

        action = torch.tanh(raw_action)

        return action, raw_action, log_prob, value, mean

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        raw_actions: torch.Tensor,
    ):
        mean, log_std = self.actor_forward(obs, update_obs_stats=False)
        value = self.critic_forward(obs, update_obs_stats=False)

        std = torch.exp(log_std).expand_as(mean)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value