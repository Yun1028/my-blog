from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# =========================================================
# CLI
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_interval", type=int, default=2000)

# training control
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--rollout", type=int, default=None)
parser.add_argument("--envs", type=int, default=None)

args_cli = parser.parse_args()

# headless에서 video를 쓰려면 enable_cameras=True 필요
app_launcher = AppLauncher(
    headless=args_cli.headless,
    enable_cameras=args_cli.video,
)
simulation_app = app_launcher.app


# =========================================================
# Imports
# =========================================================
import os
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Iterator

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from dragoon_locomotion_env_cfg import DragoonLocomotionEnvCfg
from dragoon_locomotion_env import DragoonLocomotionEnv
from dragoon_actor_critic import DragoonActorCritic, DragoonActorCriticCfg


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_policy_obs(obs) -> Dict[str, torch.Tensor]:
    """
    Isaac Lab env output -> policy obs dict 추출
    기대 형태:
      obs = {"policy": {"proprio": ..., "terrain": ...}}
    또는
      obs = {"proprio": ..., "terrain": ...}
    또는
      reset()이 (obs, info)를 반환하는 경우
    """
    if isinstance(obs, tuple):
        obs = obs[0]

    if "policy" in obs:
        obs = obs["policy"]

    return {
        "proprio": obs["proprio"],
        "terrain": obs["terrain"],
    }


def env_reset(env):
    out = env.reset()
    return extract_policy_obs(out)


def env_step(env, action: torch.Tensor):
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif len(out) == 4:
        obs, reward, done, info = out
        terminated = done
        truncated = torch.zeros_like(done, dtype=torch.bool)
    else:
        raise RuntimeError(f"Unexpected env.step output length: {len(out)}")

    obs = extract_policy_obs(obs)

    if not isinstance(reward, torch.Tensor):
        reward = torch.as_tensor(reward, device=action.device, dtype=torch.float32)
    if not isinstance(terminated, torch.Tensor):
        terminated = torch.as_tensor(terminated, device=action.device, dtype=torch.bool)
    if not isinstance(truncated, torch.Tensor):
        truncated = torch.as_tensor(truncated, device=action.device, dtype=torch.bool)

    reward = reward.float().view(-1)
    terminated = terminated.bool().view(-1)
    truncated = truncated.bool().view(-1)
    done = terminated | truncated

    return obs, reward, terminated, truncated, done, info


def atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    tanh(raw_action)=action 이므로 raw_action 복원용
    PPO update에서 evaluate_actions에 넣는다.
    """
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


# =========================================================
# Rollout Buffer
# =========================================================
class RolloutBuffer:
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        proprio_dim: int,
        terrain_shape: Tuple[int, int, int],
        action_dim: int,
        device: torch.device,
    ):
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device

        c, h, w = terrain_shape

        self.proprio = torch.zeros(horizon, num_envs, proprio_dim, device=device)
        self.terrain = torch.zeros(horizon, num_envs, c, h, w, device=device)

        self.actions = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.raw_actions = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, 1, device=device)
        self.values = torch.zeros(horizon, num_envs, 1, device=device)

        self.rewards = torch.zeros(horizon, num_envs, 1, device=device)
        self.terminated = torch.zeros(horizon, num_envs, 1, device=device)
        self.truncated = torch.zeros(horizon, num_envs, 1, device=device)
        self.dones = torch.zeros(horizon, num_envs, 1, device=device)

        self.advantages = torch.zeros(horizon, num_envs, 1, device=device)
        self.returns = torch.zeros(horizon, num_envs, 1, device=device)

        self.step = 0

    def reset(self):
        self.step = 0

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        raw_action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        done: torch.Tensor,
    ):
        t = self.step
        self.proprio[t].copy_(obs["proprio"])
        self.terrain[t].copy_(obs["terrain"])
        self.actions[t].copy_(action)
        self.raw_actions[t].copy_(raw_action)
        self.log_probs[t].copy_(log_prob)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward.unsqueeze(-1))
        self.terminated[t].copy_(terminated.float().unsqueeze(-1))
        self.truncated[t].copy_(truncated.float().unsqueeze(-1))
        self.dones[t].copy_(done.float().unsqueeze(-1))
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_terminated: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        last_gae = torch.zeros(self.num_envs, 1, device=self.device)

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_value = last_value
                next_non_terminal = 1.0 - last_terminated.float().unsqueeze(-1)
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.terminated[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def iter_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        total = self.horizon * self.num_envs

        flat_proprio = self.proprio.reshape(total, *self.proprio.shape[2:])
        flat_terrain = self.terrain.reshape(total, *self.terrain.shape[2:])
        flat_actions = self.actions.reshape(total, *self.actions.shape[2:])
        flat_raw_actions = self.raw_actions.reshape(total, *self.raw_actions.shape[2:])
        flat_log_probs = self.log_probs.reshape(total, 1)
        flat_values = self.values.reshape(total, 1)
        flat_advantages = self.advantages.reshape(total, 1)
        flat_returns = self.returns.reshape(total, 1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]

            yield {
                "obs": {
                    "proprio": flat_proprio[mb_idx],
                    "terrain": flat_terrain[mb_idx],
                },
                "actions": flat_actions[mb_idx],
                "raw_actions": flat_raw_actions[mb_idx],
                "old_log_probs": flat_log_probs[mb_idx],
                "old_values": flat_values[mb_idx],
                "advantages": flat_advantages[mb_idx],
                "returns": flat_returns[mb_idx],
            }


# =========================================================
# Train Config
# =========================================================
@dataclass
class PPOTrainCfg:
    seed: int = 42
    device: str = "cuda"

    total_env_steps: int = 20_000_00000
    rollout_steps: int = 64
    #학습률-------------------------
    learning_rate: float = 1e-4 #4e-4
    weight_decay: float = 0.0


    gamma: float = 0.96 #0.99
    gae_lambda: float = 0.95 #0.95

    ppo_epochs: int = 4
    minibatch_size: int = 16384

    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    ent_coef: float = 0.01 #0.01--
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    save_every_iters: int = 50
    log_every_iters: int = 1

    exp_name: str = "dragoon_ppo"
    ckpt_dir: str = "./checkpoints_dragoon"

    use_amp: bool = False


# =========================================================
# PPO Trainer
# =========================================================
class PPOTrainer:
    def __init__(
        self,
        env,
        model: DragoonActorCritic,
        cfg: PPOTrainCfg,
        device: torch.device,
    ):
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.model = model
        self.cfg = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        self.num_envs = self.base_env.num_envs
        self.rollout_steps = cfg.rollout_steps

        self.buffer = RolloutBuffer(
            horizon=cfg.rollout_steps,
            num_envs=self.num_envs,
            proprio_dim=66,
            terrain_shape=(1, 21, 21),
            action_dim=16,
            device=device,
        )

        self.global_env_steps = 0
        self.iteration = 0

        self.obs = env_reset(env)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def load(self, ckpt_path: str):

        payload = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        self.iteration = payload.get("iteration", 0)

        # reset steps
        self.global_env_steps = 0

        print("resume iteration:", self.iteration)

    @torch.no_grad()
    def collect_rollout(self):
        self.buffer.reset()

        reward_sum = 0.0
        done_sum = 0.0

        for _ in range(self.rollout_steps):
            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                action, raw_action, log_prob, value, _ = self.model.act(self.obs)

            next_obs, reward, terminated, truncated, done, _info = env_step(self.env, action)

            self.buffer.add(
                obs=self.obs,
                action=action,
                raw_action=raw_action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                done=done,
            )

            reward_sum += reward.mean().item()
            done_sum += done.float().mean().item()

            self.obs = next_obs
            self.global_env_steps += self.num_envs

        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            last_value = self.model.critic_forward(self.obs, update_obs_stats=False)

        last_terminated = self.buffer.terminated[self.buffer.step - 1].squeeze(-1).bool()

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_terminated=last_terminated,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        # ===============================
        # DEBUG PRINTS
        # ===============================
        if self.iteration % 10 == 0:
            print("proprio mean:", self.obs["proprio"].mean().item(),
                "std:", self.obs["proprio"].std().item())

            print("terrain mean:", self.obs["terrain"].mean().item(),
                "std:", self.obs["terrain"].std().item())

            print("adv mean:", self.buffer.advantages.mean().item(),
                "std:", self.buffer.advantages.std().item())

            print("term rate:", self.buffer.terminated.float().mean().item(),
                "trunc rate:", self.buffer.truncated.float().mean().item())

        return {
            "rollout_reward_mean": reward_sum / self.rollout_steps,
            "rollout_done_rate": done_sum / self.rollout_steps,
        }

    def ppo_update(self):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for batch in self.buffer.iter_minibatches(self.cfg.minibatch_size):
                obs = batch["obs"]
                raw_actions = batch["raw_actions"]
                old_log_probs = batch["old_log_probs"]
                old_values = batch["old_values"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    new_log_probs, entropy, new_values = self.model.evaluate_actions(obs, raw_actions)

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_coef,
                        1.0 + self.cfg.clip_coef,
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -self.cfg.value_clip_coef,
                        self.cfg.value_clip_coef,
                    )
                    value_loss_unclipped = (new_values - returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + self.cfg.vf_coef * value_loss
                        - self.cfg.ent_coef * entropy_loss
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_loss += loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_loss": total_loss / max(num_updates, 1),
        }

    def save(self, tag: str | None = None):
        name = f"iter_{self.iteration}.pt" if tag is None else f"{tag}.pt"
        path = os.path.join(self.cfg.ckpt_dir, name)

        payload = {
            "iteration": self.iteration,
            "global_env_steps": self.global_env_steps,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_cfg": self.cfg.__dict__,
        }
        torch.save(payload, path)
        return path

    def train(self):
        start_time = time.time()

        while self.global_env_steps < self.cfg.total_env_steps:
            self.iteration += 1

            rollout_stats = self.collect_rollout()
            update_stats = self.ppo_update()

            if self.iteration % self.cfg.log_every_iters == 0:
                elapsed = time.time() - start_time
                sps = self.global_env_steps / max(elapsed, 1e-6)

                print(
                    f"[Iter {self.iteration:05d}] "
                    f"steps={self.global_env_steps:,} | "
                    f"sps={sps:,.1f} | "
                    f"rew={rollout_stats['rollout_reward_mean']:.4f} | "
                    f"done={rollout_stats['rollout_done_rate']:.4f} | "
                    f"pi_loss={update_stats['policy_loss']:.4f} | "
                    f"v_loss={update_stats['value_loss']:.4f} | "
                    f"ent={update_stats['entropy']:.4f} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6g}"
                )

            if self.iteration % self.cfg.save_every_iters == 0:
                ckpt = self.save()
                print(f"saved checkpoint: {ckpt}")

        final_ckpt = self.save(tag="final")
        print(f"training finished. final checkpoint: {final_ckpt}")


# =========================================================
# Main
# =========================================================
def main():
    train_cfg = PPOTrainCfg()

    # CLI override
    if args_cli.lr is not None:
        train_cfg.learning_rate = args_cli.lr
    if args_cli.rollout is not None:
        train_cfg.rollout_steps = args_cli.rollout

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)

    # -----------------------------
    # Env
    # -----------------------------
    env_cfg = DragoonLocomotionEnvCfg()

    if args_cli.envs is not None:
        env_cfg.scene.num_envs = args_cli.envs
    else:
        env_cfg.scene.num_envs = 2048

    env_cfg.scene.env_spacing = 5.0

    if hasattr(env_cfg, "terrain") and hasattr(env_cfg.terrain, "env_spacing"):
        env_cfg.terrain.env_spacing = env_cfg.scene.env_spacing

    # viewer: 월드 고정, 시작 시점만 env0 로봇 기준으로 맞춤
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = (18.0, 0.0, 10.2)
        env_cfg.viewer.lookat = (0.0, 0.0, 1.2)
        env_cfg.viewer.origin_type = "world"

    render_mode = "rgb_array" if args_cli.video else None
    env = DragoonLocomotionEnv(env_cfg, render_mode=render_mode)
    env.seed(train_cfg.seed)

    # 처음 한 번만 env0 로봇 기준으로 카메라 맞춤
    robot_pos = env.robot.data.root_pos_w[0].cpu().numpy()
    eye = (
        robot_pos[0] + 18.0,
        robot_pos[1] + 0.0,
        robot_pos[2] + 10.2,
    )
    lookat = (
        robot_pos[0],
        robot_pos[1],
        robot_pos[2] + 1.2,
    )
    env.sim.set_camera_view(eye, lookat)

    # -----------------------------
    # Video wrapper
    # -----------------------------
    if args_cli.video:
        video_dir = os.path.join(train_cfg.ckpt_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

        print(
            f"[Video] enabled: dir={video_dir}, "
            f"capture_every_steps={args_cli.video_interval}, "
            f"length={args_cli.video_length}"
        )

    # -----------------------------
    # Model
    # -----------------------------
    model_cfg = DragoonActorCriticCfg(
        proprio_dim=66,
        action_dim=16,
        proprio_feat_dim=128,
        terrain_feat_dim=256,
        fusion_hidden_dims=(512, 512, 256),
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        init_log_std=-0.7,
        use_obs_norm=True,
    )
    model = DragoonActorCritic(model_cfg).to(device)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = PPOTrainer(
        env=env,
        model=model,
        cfg=train_cfg,
        device=device,
    )

    if args_cli.resume is not None:
        trainer.load(args_cli.resume)

    trainer.train()


if __name__ == "__main__":
    main()
    simulation_app.close()