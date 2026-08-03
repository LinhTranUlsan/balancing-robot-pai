# twip_rsl_v2_env_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from . import mdp
from .twip import TwoWheel_CFG

WHEEL_JOINT_NAMES = ["wheel1_motor1_joint", "wheel2_motor2_joint"]
MOTOR_TORQUE_MAX = 0.49  # Nm — matches the effort_limit of the "wheels" actuator in twip.py
# Train with a clean continuous action (no sim dead-zone); the real motor dead-zone is handled by
# the firmware shaper. A sim dead-zone gives no gradient for small commands and weakens the policy.
MOTOR_DEADZONE = 0.0

##
# Scene definition
##


@configclass
class TwoWheelSceneCfg(InteractiveSceneCfg):
    """Playground + TWIP robot."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
                # "max": use the larger friction of the two contacting materials (wheel/floor).
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
        ),
    )

    robot: ArticulationCfg = TwoWheel_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # IMU on the chassis, simulating a physical sensor (MPU6050, ICM42688...). lin_acc_b measures
    # R^T*(a_body + g), like a real accelerometer (unlike the clean projected_gravity = R^T*g).
    imu: ImuCfg = ImuCfg(
        # Attach to base_link (fixed joints were merged into it); z=0.129 keeps the original sensor height.
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.129),
            # +90° about Z so sensor-Y aligns with the wheels' pitch axis (X_robot).
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
        update_period=0.0,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action: ONE command in [-1, 1] -> torque, applied EQUALLY to both wheels (1-D)."""

    # Single symmetric command so (a) the exported ONNX has one output matching the firmware
    # (POLICY_ACT_DIM == 1) and (b) the policy focuses on fore/aft balancing (no steering/yaw).
    wheel_effort = mdp.SymmetricWheelEffortActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=MOTOR_TORQUE_MAX,
        deadzone=MOTOR_DEADZONE,
    )


@configclass
class ObservationsCfg:
    """Observation: only pitch angle and pitch angular rate from the IMU."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Gaussian noise models residual sensor/filter error. Pitch std kept small (0.01 rad) so the
        # tiny near-balance tilt signal (~0.03 rad) isn't buried in noise.
        pitch_angle = ObsTerm(func=mdp.imu_pitch_angle, noise=Gnoise(mean=0.0, std=0.01))
        pitch_rate = ObsTerm(func=mdp.imu_pitch_rate, noise=Gnoise(mean=0.0, std=0.02))
        # Optional wheel-encoder observation (disabled: policy uses pitch angle/rate only).
        # wheel_vel = ObsTerm(
        #     func=mdp.joint_vel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Wheel/floor friction fixed at startup to match the ground material (static=1.5, dynamic=1.2).
    wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel1", "wheel2"]),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Randomize the chassis center of mass so the policy is robust to a real-world CoM offset.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (-0.001, 0.001)},
        },
    )

    # The tilt axis is X_robot = "roll" in roll/pitch/yaw terms, so randomize the "roll" key (not "pitch").
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"roll": (-0.2, 0.2)},
            "velocity_range": {"roll": (-0.5, 0.5)},
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Random push along Y (world) every 2-4 s to teach balance recovery. Kept mild (+-1.0 m/s) so it
    # doesn't topple the robot before it has learned; raise it later for more robustness.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"y": (-1.0, 1.0)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant reward for staying alive.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty. Kept moderate (-50): a huge value blows up the PPO value function early
    # (the robot falls constantly at first) and causes "always fails".
    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # (3) Primary task: keep the chassis upright (projected_gravity_b away from [0,0,-1]).
    upright = RewTerm(
        func=mdp.base_upright_penalty,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # (3b) Bounded upright bonus, complementing the unbounded "upright" penalty.
    upright_bonus = RewTerm(
        func=mdp.base_upright_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.2},
    )
    # (4) Shaping: damp tilt oscillation.
    pitch_rate = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # (5) Yaw-rate shaping (disabled: yaw isn't observed and the symmetric action barely yaws).
    # yaw_rate = RewTerm(
    #     func=mdp.ang_vel_z_l2,
    #     weight=-0.75,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # (6) Forward-drift shaping (disabled: velocity isn't observed, and some slow drift is OK on the
    # real IMU-only robot).
    # lin_vel_x = RewTerm(
    #     func=mdp.lin_vel_x_l2,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # (7) Wheel-difference shaping (disabled with the encoder observation).
    # wheel_vel_diff = RewTerm(
    #     func=mdp.wheel_vel_diff_l2,
    #     weight=-0.0000,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )
    # (8) Shaping: smooth the action to avoid motor jerk.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # (9) Action-magnitude shaping (disabled).
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.000)
    # (10) Wheel-velocity shaping (disabled with the encoder observation).
    # wheel_vel_l2 = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )
    # (11) PD-imitation shaping (disabled: we want a pure RL policy, and its kp/kd sign is unverified).
    # pid_mimic = RewTerm(
    #     func=mdp.pid_mimic_l2,
    #     weight=-1.0,
    #     params={"kp": 8.0, "kd": 0.5, "angle_setpoint": 0.0},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Episode time out.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fell past the allowed tilt (limit_angle=0.7 rad ~40°, from projected_gravity_b).
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    )


##
# Environment configuration
##


@configclass
class TwipRslV2EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TwoWheelSceneCfg = TwoWheelSceneCfg(num_envs=8196, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # viewer settings
        self.viewer.eye = (0.8, 0.8, 0.5)
        # simulation settings
        # dt * decimation = 0.005 * 2 = 0.01s -> the system's sampling/control period is 10ms
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
