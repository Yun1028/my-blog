from __future__ import annotations

import math
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import (
    quat_apply,
    quat_rotate_inverse,
    sample_uniform,
)

from dragoon_locomotion_env_cfg import DragoonLocomotionEnvCfg


class DragoonLocomotionEnv(DirectRLEnv):
    cfg: DragoonLocomotionEnvCfg

    def __init__(self, cfg: DragoonLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self.height_scanner: RayCaster = self.scene["height_scanner"]
        self.foot_contact: ContactSensor = self.scene["foot_contact"]

        self._joint_ids = self.robot.find_joints(self.cfg.joint_names)[0]
        self._feet_ids = self.robot.find_bodies(self.cfg.foot_link_names)[0]
        self._base_id = self.robot.find_bodies([self.cfg.base_link_name])[0][0]

        # command buffers (world frame)
        self.move_cmd = torch.zeros((self.num_envs, 2), device=self.device)
        self.look_cmd = torch.zeros((self.num_envs, 2), device=self.device)
        self.target_height = torch.zeros((self.num_envs,), device=self.device)
        self.default_base_height = torch.zeros((self.num_envs,), device=self.device)

        # mode buffer
        self.is_stop_cmd = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # action buffers
        self.actions = torch.zeros((self.num_envs, self.cfg.action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        # cached tensors
        self._forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
        self._up_world = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)

        # local foot z-axis
        self._foot_z_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # default joint pose
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self._joint_ids].clone()
        self.default_joint_vel = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)

        # look direction / speed resampling schedule
        self.look_cmd_resample_interval_steps = int(10.0 / self.step_dt)
        self.next_look_cmd_resample_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # 누적 균형용 버퍼
        num_feet = len(self._feet_ids)
        self.foot_contact_time_accum = torch.zeros((self.num_envs, num_feet), device=self.device)
        self.foot_push_accum = torch.zeros((self.num_envs, num_feet), device=self.device)

        self.stop_vel_xy_ema = torch.zeros((self.num_envs, 2), device=self.device)
        self.stop_ang_xy_ema = torch.zeros((self.num_envs, 2), device=self.device)
        
        # episode statistics buffers
        self._init_episode_stats_buffers()

        print("robot count:", self.robot.num_instances)
        print("env num:", self.num_envs)
        print("default_root_state shape:", self.robot.data.default_root_state.shape)
        print("feet ids:", self._feet_ids)
        print("foot names:", self.cfg.foot_link_names)

    def _init_episode_stats_buffers(self):
        self.ep_len = torch.zeros(self.num_envs, device=self.device)
        self.ep_reward_sum = torch.zeros(self.num_envs, device=self.device)

        self.ep_r_move = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_stop_still = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_height = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_upright = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_heading = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_alive = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_foot_vertical = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_contact_time_balance = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_force_balance = torch.zeros(self.num_envs, device=self.device)
        self.ep_r_contact_count = torch.zeros(self.num_envs, device=self.device)

        self.ep_p_action_rate = torch.zeros(self.num_envs, device=self.device)
        self.ep_p_foot_slip = torch.zeros(self.num_envs, device=self.device)
        self.ep_p_fall = torch.zeros(self.num_envs, device=self.device)

    def _accumulate_episode_stats(
        self,
        reward: torch.Tensor,
        r_move: torch.Tensor,
        r_stop_still: torch.Tensor,
        r_height: torch.Tensor,
        r_upright: torch.Tensor,
        r_heading: torch.Tensor,
        r_alive: torch.Tensor,
        r_foot_vertical: torch.Tensor,
        r_contact_time_balance: torch.Tensor,
        r_force_balance: torch.Tensor,
        r_contact_count: torch.Tensor,
        p_action_rate: torch.Tensor,
        p_foot_slip: torch.Tensor,
        p_fall: torch.Tensor,
    ):
        self.ep_len += 1.0
        self.ep_reward_sum += reward

        self.ep_r_move += r_move
        self.ep_r_stop_still += r_stop_still
        self.ep_r_height += r_height
        self.ep_r_upright += r_upright
        self.ep_r_heading += r_heading
        self.ep_r_alive += r_alive
        self.ep_r_foot_vertical += r_foot_vertical
        self.ep_r_contact_time_balance += r_contact_time_balance
        self.ep_r_force_balance += r_force_balance
        self.ep_r_contact_count += r_contact_count

        self.ep_p_action_rate += p_action_rate
        self.ep_p_foot_slip += p_foot_slip
        self.ep_p_fall += p_fall

    def _print_episode_stats_env0(self, died: torch.Tensor, time_out: torch.Tensor):
        env_id = 0
        if not (died[env_id] or time_out[env_id]):
            return

        ep_len = max(self.ep_len[env_id].item(), 1.0)

        avg_reward = (self.ep_reward_sum[env_id] / ep_len).item()

        avg_r_move = (self.ep_r_move[env_id] / ep_len).item()
        avg_r_stop_still = (self.ep_r_stop_still[env_id] / ep_len).item()
        avg_r_height = (self.ep_r_height[env_id] / ep_len).item()
        avg_r_upright = (self.ep_r_upright[env_id] / ep_len).item()
        avg_r_heading = (self.ep_r_heading[env_id] / ep_len).item()
        avg_r_alive = (self.ep_r_alive[env_id] / ep_len).item()
        avg_r_foot_vertical = (self.ep_r_foot_vertical[env_id] / ep_len).item()
        avg_r_contact_time_balance = (self.ep_r_contact_time_balance[env_id] / ep_len).item()
        avg_r_force_balance = (self.ep_r_force_balance[env_id] / ep_len).item()
        avg_r_contact_count = (self.ep_r_contact_count[env_id] / ep_len).item()

        avg_p_action_rate = (self.ep_p_action_rate[env_id] / ep_len).item()
        avg_p_foot_slip = (self.ep_p_foot_slip[env_id] / ep_len).item()
        avg_p_fall = (self.ep_p_fall[env_id] / ep_len).item()

        print("========== EPISODE SUMMARY ENV0 ==========")
        print("env id                     :", env_id)
        print("done by died               :", bool(died[env_id].item()))
        print("done by time_out           :", bool(time_out[env_id].item()))
        print("episode length             :", ep_len)
        print("avg total reward           :", avg_reward)

        print("avg r_move                 :", avg_r_move)
        print("avg r_stop_still           :", avg_r_stop_still)
        print("avg r_height               :", avg_r_height)
        print("avg r_upright              :", avg_r_upright)
        print("avg r_heading              :", avg_r_heading)
        print("avg r_alive                :", avg_r_alive)
        print("avg r_foot_vertical        :", avg_r_foot_vertical)
        print("avg r_contact_time_balance :", avg_r_contact_time_balance)
        print("avg r_force_balance        :", avg_r_force_balance)
        print("avg r_contact_count        :", avg_r_contact_count)

        print("avg p_action_rate          :", avg_p_action_rate)
        print("avg p_foot_slip            :", avg_p_foot_slip)
        print("avg p_fall                 :", avg_p_fall)

        print("env0 foot_contact_time_accum:", self.foot_contact_time_accum[0].detach().cpu().numpy())
        print("env0 foot_push_accum        :", self.foot_push_accum[0].detach().cpu().numpy())
        print("==========================================")

    def _reset_episode_stats(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.ep_len[env_ids] = 0.0
        self.ep_reward_sum[env_ids] = 0.0

        self.ep_r_move[env_ids] = 0.0
        self.ep_r_stop_still[env_ids] = 0.0
        self.ep_r_height[env_ids] = 0.0
        self.ep_r_upright[env_ids] = 0.0
        self.ep_r_heading[env_ids] = 0.0
        self.ep_r_alive[env_ids] = 0.0
        self.ep_r_foot_vertical[env_ids] = 0.0
        self.ep_r_contact_time_balance[env_ids] = 0.0
        self.ep_r_force_balance[env_ids] = 0.0
        self.ep_r_contact_count[env_ids] = 0.0

        self.ep_p_action_rate[env_ids] = 0.0
        self.ep_p_foot_slip[env_ids] = 0.0
        self.ep_p_fall[env_ids] = 0.0

    #
    # scene
    #
    def _setup_scene(self):
        self.scene = InteractiveScene(self.cfg.scene)

        self.cfg.terrain.class_type(self.cfg.terrain)

        robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = robot

        self.scene.sensors["height_scanner"] = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["foot_contact"] = ContactSensor(self.cfg.foot_contact)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.8, 0.8, 0.8),
        )
        dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

        distant_light_cfg = sim_utils.DistantLightCfg(
            intensity=2500.0,
            color=(1.0, 1.0, 1.0),
        )
        distant_light_cfg.func("/World/DistantLight", distant_light_cfg)

    #
    # action application
    #
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(torch.clamp(actions, -1.0, 1.0))

        q_target = self.default_joint_pos + self.cfg.action_scale * self.actions

        q_low = self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        q_high = self.robot.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        q_target = torch.clamp(q_target, q_low, q_high)

        self.robot.set_joint_position_target(q_target, joint_ids=self._joint_ids)

    def _apply_action(self):
        pass

    #
    # helper: look_cmd resampling
    #
    def _maybe_resample_look_commands(self):
        cur_step = self.episode_length_buf

        env_ids = torch.nonzero(cur_step >= self.next_look_cmd_resample_step, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        self._resample_commands(env_ids)

    #
    # observations
    #
    def _get_observations(self) -> dict:
        self._maybe_resample_look_commands()

        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        base_lin_vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
        base_ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity_b = quat_rotate_inverse(root_quat_w, gravity_w)

        move_cmd_3 = torch.cat(
            [self.move_cmd, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )
        move_cmd_b = quat_rotate_inverse(root_quat_w, move_cmd_3)[:, :2]

        look_cmd_3 = torch.cat(
            [self.look_cmd, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )
        look_cmd_b = quat_rotate_inverse(root_quat_w, look_cmd_3)[:, :2]

        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids]
        joint_pos_offset = joint_pos - self.default_joint_pos

        net_forces = self.foot_contact.data.net_forces_w
        foot_force_norm = torch.norm(net_forces, dim=-1)
        foot_contact = (foot_force_norm > self.cfg.contact_force_threshold).float()

        proprio = torch.cat(
            [
                move_cmd_b,
                look_cmd_b,
                self.target_height.unsqueeze(-1),
                base_lin_vel_b,
                base_ang_vel_b,
                projected_gravity_b,
                joint_pos_offset,
                joint_vel,
                self.prev_actions,
                foot_contact,
            ],
            dim=-1,
        )

        ray_hits_w = self.height_scanner.data.ray_hits_w
        base_pos_w = self.robot.data.root_pos_w[:, :3].unsqueeze(1)

        rel_h = base_pos_w[..., 2] - ray_hits_w[..., 2]
        terrain = rel_h.view(self.num_envs, 1, 21, 21)

        return {
            "policy": {
                "proprio": proprio,
                "terrain": terrain,
            }
        }

    #
    # rewards
    #
    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        abs_height = root_pos_w[:, 2]

        base_lin_vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
        base_ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        vel_xy = base_lin_vel_b[:, :2]

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity_b = quat_rotate_inverse(root_quat_w, gravity_w)

        move_cmd_3 = torch.cat(
            [self.move_cmd, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )
        move_cmd_b = quat_rotate_inverse(root_quat_w, move_cmd_3)[:, :2]

        # -------------------------------------------------
        # cfg / fallback defaults
        # -------------------------------------------------
        rew_foot_vertical = getattr(self.cfg, "rew_foot_vertical", 0.3)
        rew_contact_time_balance = getattr(self.cfg, "rew_contact_time_balance", 0.2)
        rew_force_balance = getattr(self.cfg, "rew_force_balance", 0.2)
        rew_contact_count = getattr(self.cfg, "rew_contact_count", 0.25)
        rew_stop_still = getattr(self.cfg, "rew_stop_still", 0.4)

        foot_vertical_angle_scale = getattr(self.cfg, "foot_vertical_angle_scale", 10.0)
        contact_time_balance_scale = getattr(self.cfg, "contact_time_balance_scale", 10.0)
        force_balance_scale = getattr(self.cfg, "force_balance_scale", 10.0)

        balance_warmup_time = getattr(self.cfg, "balance_warmup_time", 0.5)
        push_force_sqrt_eps = getattr(self.cfg, "push_force_sqrt_eps", 1e-6)

        stop_speed_eps = getattr(self.cfg, "stop_speed_eps", 1e-4)
        stop_still_vel_scale = getattr(self.cfg, "stop_still_vel_scale", 8.0)
        stop_still_ang_scale = getattr(self.cfg, "stop_still_ang_scale", 2.0)
    
        # stop mode 진입용 각도: cfg 안 쓰고 여기 고정
        stop_heading_deg = 10.0
        stop_heading_cos_thresh = math.cos(math.radians(stop_heading_deg))

        # -------------------------------------------------
        # 1) contact / slip / push / support feet
        # -------------------------------------------------
        net_forces = self.foot_contact.data.net_forces_w                      # [N, 4, 3]
        foot_force_norm = torch.norm(net_forces, dim=-1)                     # [N, 4]
        foot_contact = (foot_force_norm > self.cfg.contact_force_threshold).float()

        contact_count = foot_contact.sum(dim=-1)
        contact_count_safe = torch.clamp(contact_count, min=1.0)

        foot_pos_w = self.robot.data.body_pos_w[:, self._feet_ids, :]        # [N, 4, 3]

        # foot origin -> ground 근사
        foot_ground_offset = getattr(self.cfg, "foot_ground_offset", 0.9)
        support_foot_ground_z = foot_pos_w[..., 2] - foot_ground_offset
        support_z_mean = (support_foot_ground_z * foot_contact).sum(dim=-1) / contact_count_safe

        foot_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_ids, :]
        foot_speed_xy_sq = torch.sum(foot_vel_w[..., :2] ** 2, dim=-1)
        p_foot_slip = torch.sum(foot_contact * foot_speed_xy_sq, dim=-1)

        foot_push_z = torch.clamp(net_forces[..., 2], min=0.0)
        foot_push_processed = torch.sqrt(foot_push_z + push_force_sqrt_eps) * foot_contact

        self.foot_contact_time_accum += foot_contact * self.step_dt
        self.foot_push_accum += foot_push_processed * self.step_dt

        # -------------------------------------------------
        # 2) move tracking reward raw
        # -------------------------------------------------
        vel_err = vel_xy - move_cmd_b
        r_move_raw = torch.exp(-2.0 * torch.sum(vel_err * vel_err, dim=-1))

        # -------------------------------------------------
        # 3) height reward
        # support feet 평균 높이 기준
        # contact 없으면 중앙 ray fallback
        # -------------------------------------------------
        ray_hits_w = self.height_scanner.data.ray_hits_w
        center_idx = (21 * 21) // 2
        ground_z_center = ray_hits_w[:, center_idx, 2]
        ground_z_center = torch.nan_to_num(ground_z_center, nan=1e-4, posinf=1e-4, neginf=1e-4)

        support_ground_z = torch.where(contact_count > 0.0, support_z_mean, ground_z_center)
        rel_height = root_pos_w[:, 2] - support_ground_z
        height_err = rel_height - self.target_height
        r_height = torch.exp(-7.0 * height_err * height_err)

        # -------------------------------------------------
        # 4) upright reward
        # -------------------------------------------------
        tilt_mag = torch.norm(projected_gravity_b[:, :2], dim=-1)
        r_upright = torch.exp(-self.cfg.upright_err_scale * tilt_mag * tilt_mag)

        # -------------------------------------------------
        # 5) heading reward
        # -------------------------------------------------
        base_forward_w = quat_apply(root_quat_w, self._forward_local)
        base_forward_xy = base_forward_w[:, :2]
        base_forward_xy = base_forward_xy / (torch.norm(base_forward_xy, dim=-1, keepdim=True) + 1e-6)

        look_cmd_xy = self.look_cmd
        look_cmd_norm = torch.norm(look_cmd_xy, dim=-1, keepdim=True)
        look_dir_xy = look_cmd_xy / (look_cmd_norm + 1e-6)

        heading_cos = torch.sum(base_forward_xy * look_dir_xy, dim=-1)
        heading_cos = torch.clamp(heading_cos, -1.0, 1.0)

        r_heading = 0.5 * (heading_cos + 1.0)
        r_heading = r_heading ** 2

        heading_active = (look_cmd_norm.squeeze(-1) > 1e-4).float()
        r_heading = heading_active * r_heading + (1.0 - heading_active)

        # -------------------------------------------------
        # 6) mode split (move / stop)
        # 규칙:
        # - speed > 0  -> move
        # - speed == 0 but heading not aligned -> move
        # - speed == 0 and heading aligned     -> stop
        # -------------------------------------------------
        move_speed_world = torch.norm(self.move_cmd, dim=-1)
        cmd_stop = move_speed_world <= stop_speed_eps
        heading_aligned_for_stop = heading_cos >= stop_heading_cos_thresh

        is_stop = cmd_stop & heading_aligned_for_stop
        is_move = ~is_stop

        is_stop_f = is_stop.float()
        is_move_f = is_move.float()

        # move reward는 move mode에서만
        move_reward_gate = (heading_cos >= math.cos(math.radians(20.0))).float()
        r_move = r_move_raw * move_reward_gate * is_move_f

        # -------------------------------------------------
        # 7) stop still reward
        # 순간 속도 대신 EMA + deadband 사용
        # -------------------------------------------------
        stop_ema_alpha = getattr(self.cfg, "stop_ema_alpha", 0.15)
        stop_vel_deadband = getattr(self.cfg, "stop_vel_deadband", 0.08)
        stop_ang_deadband = getattr(self.cfg, "stop_ang_deadband", 0.08)

        # EMA 업데이트
        self.stop_vel_xy_ema = (1.0 - stop_ema_alpha) * self.stop_vel_xy_ema + stop_ema_alpha * vel_xy
        self.stop_ang_xy_ema = (1.0 - stop_ema_alpha) * self.stop_ang_xy_ema + stop_ema_alpha * base_ang_vel_b[:, :2]

        # norm 계산
        stop_speed_ema = torch.norm(self.stop_vel_xy_ema, dim=-1)
        stop_ang_speed_ema = torch.norm(self.stop_ang_xy_ema, dim=-1)

        # deadband 이하는 0 처리
        stop_speed_eff = torch.clamp(stop_speed_ema - stop_vel_deadband, min=0.0)
        stop_ang_speed_eff = torch.clamp(stop_ang_speed_ema - stop_ang_deadband, min=0.0)

        # penalty
        stop_vel_pen = stop_speed_eff * stop_speed_eff
        stop_ang_pen = stop_ang_speed_eff * stop_ang_speed_eff

        r_stop_still_raw = torch.exp(
            -stop_still_vel_scale * stop_vel_pen
            -stop_still_ang_scale * stop_ang_pen
        )
        r_stop_still = r_stop_still_raw * is_stop_f

        # -------------------------------------------------
        # 8) alive reward
        # -------------------------------------------------
        r_alive = torch.ones_like(r_move)

        # -------------------------------------------------
        # 9) action rate penalty
        # -------------------------------------------------
        p_action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)

        # -------------------------------------------------
        # 10) foot vertical reward (stance only)
        # 15도에서 대략 0.5
        # -------------------------------------------------
        foot_quat_w = self.robot.data.body_quat_w[:, self._feet_ids, :]
        num_feet = foot_quat_w.shape[1]

        foot_z_local = self._foot_z_local.view(1, 1, 3).repeat(self.num_envs, num_feet, 1)
        foot_z_world = quat_apply(
            foot_quat_w.reshape(-1, 4),
            foot_z_local.reshape(-1, 3),
        ).view(self.num_envs, num_feet, 3)

        foot_vertical_cos = foot_z_world[..., 2].clamp(-1.0, 1.0)
        foot_angle = torch.acos(foot_vertical_cos)

        r_foot_vertical_each = torch.exp(-foot_vertical_angle_scale * foot_angle * foot_angle)
        r_foot_vertical_contact = (
            (foot_contact * r_foot_vertical_each).sum(dim=-1) / contact_count_safe
        )
        r_foot_vertical = torch.where(
            contact_count > 0.0,
            r_foot_vertical_contact,
            torch.ones_like(r_move),
        )

        # -------------------------------------------------
        # 11) 누적 접지시간 균형 보상
        # -------------------------------------------------
        ct_total = torch.sum(self.foot_contact_time_accum, dim=-1, keepdim=True)
        ct_ready = (ct_total.squeeze(-1) > balance_warmup_time)

        ct_share = self.foot_contact_time_accum / (ct_total + 1e-6)
        ct_target = 1.0 / self.foot_contact_time_accum.shape[1]

        ct_var = torch.mean((ct_share - ct_target) ** 2, dim=-1)
        r_contact_time_balance_raw = torch.exp(-contact_time_balance_scale * ct_var)

        r_contact_time_balance = torch.where(
            ct_ready,
            r_contact_time_balance_raw,
            torch.ones_like(r_move),
        )

        # -------------------------------------------------
        # 12) 누적 힘 분배 균형 보상
        # -------------------------------------------------
        push_total = torch.sum(self.foot_push_accum, dim=-1, keepdim=True)
        push_ready = (push_total.squeeze(-1) > balance_warmup_time)

        push_share = self.foot_push_accum / (push_total + 1e-6)
        push_target = 1.0 / self.foot_push_accum.shape[1]

        push_var = torch.mean((push_share - push_target) ** 2, dim=-1)
        r_force_balance_raw = torch.exp(-force_balance_scale * push_var)

        r_force_balance = torch.where(
            push_ready,
            r_force_balance_raw,
            torch.ones_like(r_move),
        )

        # -------------------------------------------------
        # 13) contact count reward
        # move mode:
        #   2개 이상 좋음, 3개 거의 최대, 4개 낮음
        # stop mode:
        #   4개 최대
        # -------------------------------------------------
        r_contact_count_move = torch.where(
            contact_count <= 1.0, torch.full_like(contact_count, 0.77),
            torch.where(
                contact_count <= 2.0, torch.full_like(contact_count, 0.91),
                torch.where(
                    contact_count <= 3.0, torch.full_like(contact_count, 1.00),
                    torch.full_like(contact_count, 0.25),
                ),
            ),
        )

        r_contact_count_stop = torch.where(
            contact_count <= 1.0, torch.full_like(contact_count, 0.30),
            torch.where(
                contact_count <= 2.0, torch.full_like(contact_count, 0.40),
                torch.where(
                    contact_count <= 3.0, torch.full_like(contact_count, 0.75),
                    torch.full_like(contact_count, 1.00),
                ),
            ),
        )

        r_contact_count = is_move_f * r_contact_count_move + is_stop_f * r_contact_count_stop

        # -------------------------------------------------
        # 14) fall penalty
        # -------------------------------------------------
        max_tilt = math.sin(self.cfg.max_roll_pitch_rad)

        body_too_low_abs = abs_height < self.cfg.min_root_height_abs
        body_too_high_abs = abs_height > self.cfg.max_root_height_abs

        fell = body_too_low_abs | body_too_high_abs | (tilt_mag > max_tilt)
        p_fall = fell.float() * self.cfg.pen_fall

        # -------------------------------------------------
        # total reward
        # -------------------------------------------------
        reward = (
            self.cfg.rew_move * r_move
            + rew_stop_still * r_stop_still
            + self.cfg.rew_height * r_height
            + self.cfg.rew_upright * r_upright
            + self.cfg.rew_alive * r_alive
            + self.cfg.rew_heading * r_heading
            + rew_foot_vertical * r_foot_vertical
            + rew_contact_time_balance * r_contact_time_balance
            + rew_force_balance * r_force_balance
            + rew_contact_count * r_contact_count
            - self.cfg.pen_action_rate * p_action_rate
            - self.cfg.pen_foot_slip * p_foot_slip
            - p_fall
        )

        self._accumulate_episode_stats(
            reward=reward,
            r_move=r_move,
            r_stop_still=r_stop_still,
            r_height=r_height,
            r_upright=r_upright,
            r_heading=r_heading,
            r_alive=r_alive,
            r_foot_vertical=r_foot_vertical,
            r_contact_time_balance=r_contact_time_balance,
            r_force_balance=r_force_balance,
            r_contact_count=r_contact_count,
            p_action_rate=p_action_rate,
            p_foot_slip=p_foot_slip,
            p_fall=p_fall,
        )

        if self.common_step_counter % 100 == 0:
            heading_deg0 = math.degrees(torch.acos(torch.clamp(heading_cos[0], -1.0, 1.0)).item())

            print("------ REWARD DEBUG ------")
            print("env0 abs_height              :", abs_height[0].item())
            print("env0 rel_height              :", rel_height[0].item())
            print("env0 support_ground_z        :", support_ground_z[0].item())
            print("env0 move_cmd body           :", move_cmd_b[0].detach().cpu().numpy())
            print("env0 move_cmd world          :", self.move_cmd[0].detach().cpu().numpy())
            print("env0 move_speed world        :", move_speed_world[0].item())
            
            print("env0 r_move---               :", r_move[0].item())
            print("env0 r_stop_still            :", r_stop_still[0].item())
            print("env0 r_height                :", r_height[0].item())
            print("env0 r_upright               :", r_upright[0].item())
            print("env0 r_heading---            :", r_heading[0].item())
            print("env0 r_foot_vertical         :", r_foot_vertical[0].item())
            print("env0 r_contact_time_balance  :", r_contact_time_balance[0].item())
            print("env0 r_force_balance         :", r_force_balance[0].item())
            print("env0 r_contact_count         :", r_contact_count[0].item())

            print("env0 move_gate               :", move_reward_gate[0].item())

            print("env0 p_action_rate           :", p_action_rate[0].item())
            print("env0 p_foot_slip             :", p_foot_slip[0].item())

            print("env0 contact_count           :", contact_count[0].item())
            print("env0 foot_contact            :", foot_contact[0].detach().cpu().numpy())

            print("env0 foot_push_z             :", foot_push_z[0].detach().cpu().numpy())
            print("env0 foot_push_processed     :", foot_push_processed[0].detach().cpu().numpy())

            print("env0 foot_contact_time_accum :", self.foot_contact_time_accum[0].detach().cpu().numpy())
            print("env0 foot_push_accum         :", self.foot_push_accum[0].detach().cpu().numpy())

            print("env0 ct_share                :", ct_share[0].detach().cpu().numpy())
            print("env0 push_share              :", push_share[0].detach().cpu().numpy())
            print("env0 ct_var                  :", ct_var[0].item())
            print("env0 push_var                :", push_var[0].item())

            print("env0 body_too_low            :", body_too_low_abs[0].item())
            print("env0 body_too_high           :", body_too_high_abs[0].item())
            print("env0 fell                    :", fell[0].item())

        return reward

    #
    # terminations
    #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        abs_height = root_pos_w[:, 2]

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity_b = quat_rotate_inverse(root_quat_w, gravity_w)

        tilt_mag = torch.norm(projected_gravity_b[:, :2], dim=-1)
        max_tilt = math.sin(self.cfg.max_roll_pitch_rad)

        body_too_low_abs = abs_height < self.cfg.min_root_height_abs
        body_too_high_abs = abs_height > self.cfg.max_root_height_abs
        bad_tilt = tilt_mag > max_tilt

        died = body_too_low_abs | body_too_high_abs | bad_tilt
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        self._print_episode_stats_env0(died, time_out)

        if died[0] or time_out[0]:
            print("---- DONE ENV0 ----")
            print("abs_height[0]:", abs_height[0].item())
            print("min_root_height_abs:", self.cfg.min_root_height_abs)
            print("max_root_height_abs:", self.cfg.max_root_height_abs)
            print("body_too_low_abs[0]:", body_too_low_abs[0].item())
            print("body_too_high_abs[0]:", body_too_high_abs[0].item())
            print("tilt_mag[0]:", tilt_mag[0].item())
            print("max_tilt:", max_tilt)
            print("bad_tilt[0]:", bad_tilt[0].item())
            print("died[0]:", died[0].item())
            print("time_out[0]:", time_out[0].item())
            print("episode_length_buf[0]:", self.episode_length_buf[0].item())

        return died, time_out

    #
    # reset
    #
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor):
        self.stop_vel_xy_ema[env_ids] = 0.0
        self.stop_ang_xy_ema[env_ids] = 0.0
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._reset_episode_stats(env_ids)

        self.foot_contact_time_accum[env_ids] = 0.0
        self.foot_push_accum[env_ids] = 0.0
        self.is_stop_cmd[env_ids] = False

        # random spawn yaw
        yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), device=self.device)

        half = yaw * 0.5
        q = torch.zeros((len(env_ids), 4), device=self.device)
        q[:, 0] = torch.cos(half)
        q[:, 1] = 0.0
        q[:, 2] = 0.0
        q[:, 3] = torch.sin(half)

        # root state
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 3:7] = q
        root_state[:, 2] = 1.8

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # joints
        joint_pos = self.default_joint_pos[env_ids].clone()
        joint_vel = self.default_joint_vel[env_ids].clone()

        joint_pos += sample_uniform(-0.03, 0.03, joint_pos.shape, device=self.device)

        q_low = self.robot.data.soft_joint_pos_limits[env_ids][:, self._joint_ids, 0]
        q_high = self.robot.data.soft_joint_pos_limits[env_ids][:, self._joint_ids, 1]
        joint_pos = torch.clamp(joint_pos, q_low, q_high)

        self.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids, joint_ids=self._joint_ids
        )

        # clear action buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        # optional reset nominal height cache
        ray_hits_w = self.height_scanner.data.ray_hits_w
        center_idx = (21 * 21) // 2

        ground_z = ray_hits_w[env_ids, center_idx, 2]
        ground_z = torch.nan_to_num(
            ground_z,
            nan=1e-4,
            posinf=1e-4,
            neginf=1e-4
        )

        base_z = self.robot.data.root_pos_w[env_ids, 2]
        self.default_base_height[env_ids] = base_z - ground_z

        self._resample_commands(env_ids)

    #
    # commands
    #
    def _resample_commands(self, env_ids: torch.Tensor):
        # stop / move mode
        stop_prob = getattr(self.cfg, "stop_prob", 0.3)

        stop_mask = torch.rand(len(env_ids), device=self.device) < stop_prob
        move_mask = ~stop_mask

        self.is_stop_cmd[env_ids] = stop_mask

        # stop mode -> speed = 0
        if stop_mask.any():
            stop_env_ids = env_ids[stop_mask]
            self.move_cmd[stop_env_ids, 0] = 0.0
            self.move_cmd[stop_env_ids, 1] = 0.0

        # move mode -> speed in [1.5, 2.7], direction = +X
        if move_mask.any():
            move_env_ids = env_ids[move_mask]
            speed = sample_uniform(1.5, 2.7, (len(move_env_ids),), device=self.device)
            self.move_cmd[move_env_ids, 0] = speed
            self.move_cmd[move_env_ids, 1] = 0.0

        # randomized look direction
        look_yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), device=self.device)

        self.look_cmd[env_ids, 0] = torch.cos(look_yaw)
        self.look_cmd[env_ids, 1] = torch.sin(look_yaw)

        # target body height
        self.target_height[env_ids] = 1.65

        # next resample
        self.next_look_cmd_resample_step[env_ids] = (
            self.episode_length_buf[env_ids] + self.look_cmd_resample_interval_steps
        )