from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
@configclass
class DragoonLocomotionEnvCfg(DirectRLEnvCfg):
    """Direct RL config for dragoon locomotion."""

    # -----------------------------
    # env
    # -----------------------------
    episode_length_s = 70.0
    decimation = 4
    action_scale = 0.7
    action_dim = 16

    # proprio obs:
    # command 5 + base 9 + joints 32 + prev_action 16 + foot_contact 4 = 66
    observation_space = {
    "policy": {
        "proprio": 66,
        "terrain": (1, 21, 21),
    }
    }
    action_space = 16

    # -----------------------------
    # simulation
    # -----------------------------
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,#마찰-------------------------------
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**22,
        ),
    )

    # -----------------------------
    # scene
    # -----------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=8.0,
        replicate_physics=True,
    )

    # -----------------------------
    # terrain
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.0,   # scene.env_spacing과 같은 값
    )

    # -----------------------------
    # robot
    # TODO: replace usd_path with your actual robot USD
    # activate_contact_sensors must be True for foot contacts
    # -----------------------------
    robot: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/dragoon1",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jeyun/IsaacLab/dragoon23/dragoon3/dragoon1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.6),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "Ja": 0.0, "Jb": 0.8, "Jc": -0.47, "Jd": 0.0,
            "Ja1": 0.0, "Jb1": 0.8, "Jc1": -0.47, "Jd1": 0.0,
            "Ja2": 0.0, "Jb2": 0.8, "Jc2": -0.47, "Jd2": 0.0,
            "Ja3": 0.0, "Jb3": 0.8, "Jc3": -0.47, "Jd3": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "dragoon_legs": ImplicitActuatorCfg(
            joint_names_expr=["J.*"],
            stiffness=400.0,
            damping=30.0,
            effort_limit=800.0,
            velocity_limit_sim=4.0,
        ),
    },
)

    # -----------------------------
    # sensors
    # -----------------------------
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/dragoon1/dragoon1/body_link",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.18,         # ~3.6 / 20
            size=(3.6, 3.6),         # [-1.8, 1.8] x [-1.8, 1.8]
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=5.5,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    foot_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/dragoon1/dragoon1/(leg_link3|leg0013_link|leg0023_link|leg0033_link)",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # -----------------------------
    # commands
    # -----------------------------
    move_vel_x_range = (-0.45, 0.45)
    move_vel_y_range = (-0.45, 0.45)
    height_offset_range = (0.0, 0.0)

    # -----------------------------
    # reward scales
    # -----------------------------
    rew_move = 3.0
    rew_height = 1.0
    rew_upright = 1.0
    upright_err_scale = 5.0
    rew_alive = 0.5
    rew_heading = 2.0

    pen_action_rate = 0.01
    pen_foot_slip = 0.02

    pen_fall = 3.5
    
    rew_look = 0.0
    rew_foot_flat = 0.0
    pen_energy = 0.0
    pen_body_collision = 0.0
    pen_still = 0.0

    # -----------------------------
    # new reward weights
    # -----------------------------
    rew_foot_vertical = 0.5 #0.5
    rew_contact_time_balance = 0.2 #자세 0.2, 0.2
    rew_force_balance = 0.2

    # -----------------------------
    # foot vertical reward shaping
    # exp(-scale * angle^2)
    # 15도에서 대략 0.5가 되게 하려면 10 전후
    # -----------------------------
    foot_vertical_angle_scale = 10.0

    # -----------------------------
    # accumulated balance reward shaping
    # exp(-scale * variance)
    # -----------------------------
    contact_time_balance_scale = 50.0
    force_balance_scale = 50.0

    # -----------------------------
    # accumulated balance warmup
    # 누적량이 너무 적을 때는 중립 보상(1.0)
    # -----------------------------
    balance_warmup_time = 2.5

    # -----------------------------
    # push accumulation preprocessing
    # sqrt(push_z + eps)
    # 큰 순간 force 스파이크 완화
    # -----------------------------
    push_force_sqrt_eps = 1e-6

    rew_contact_count = 0.15#자세 0.15
    rew_stop_still = 0.4

    stop_prob = 0.3
    stop_speed_eps = 1e-4
    stop_still_vel_scale = 8.0
    stop_still_ang_scale = 2.0

    stop_ema_alpha = 0.15 #stop 속도 측정을 위한것
    stop_vel_deadband = 0.08
    stop_ang_deadband = 0.08
    # -----------------------------
    # termination
    # -----------------------------
    max_roll_pitch_rad = math.radians(20.0)
    contact_force_threshold = 5.0

    min_root_height_abs = 1.55
    max_root_height_abs = 1.9

    # -----------------------------
    # misc
    # -----------------------------
    joint_names = [
        "Ja", "Jb", "Jc", "Jd",
        "Ja1", "Jb1", "Jc1", "Jd1",
        "Ja2", "Jb2", "Jc2", "Jd2",
        "Ja3", "Jb3", "Jc3", "Jd3",
    ]

    foot_link_names = [
        "leg_link3",
        "leg0013_link",
        "leg0023_link",
        "leg0033_link",
    ]

    base_link_name = "body_link"