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
# >>> CHANGED: 40/255 (=0.157) -> 0.0. A large sim dead-zone creates a flat "zero-torque" band that
# gives the policy no gradient for small commands, so it collapsed to weak sub-threshold outputs.
# The real motor dead-zone is handled on the FIRMWARE side (pulse-density + min-duty shaper), so we
# train with a clean continuous action. Raise this again only if you specifically want to model it.
MOTOR_DEADZONE = 0.0  # was 40/255; real dead-zone handled by the firmware motor shaper

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
                # "max": always take the larger friction of the two contacting materials (wheel - floor),
                # avoiding a reduction from the default "average" combine mode when one side has lower friction.
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

    # IMU mounted on the chassis, simulating a physical sensor (MPU6050, ICM42688...)
    # lin_acc_b = R^T * (a_body + gravity_bias) — exactly what a real accelerometer measures,
    # unlike projected_gravity = R^T * g (a perfect measurement, no noise from chassis acceleration)
    imu: ImuCfg = ImuCfg(
        # merge_fixed_joints=True (twip.py) merges upper_base_link into base_link, so it must attach to base_link;
        # add 0.079 (z offset of the old upper_base_joint) to keep the sensor at the height of the original design.
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.129),
            # +90° about Z: Z_sensor = Z_robot (unchanged, "up" stays the same)
            # X_sensor = -Y_robot, Y_sensor = X_robot (the wheels' true pitch axis → sensor-Y)
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

    # >>> CHANGED: was JointEffortActionWithDeadzoneCfg (2-D, one effort per wheel).
    # Now a single symmetric command so that: (a) the exported ONNX has ONE output, matching the
    # on-board firmware (POLICY_ACT_DIM == 1); (b) the policy cannot steer/yaw, so training focuses
    # on pure fore/aft balancing. The action term clamps torque to [-scale, scale] internally, so
    # the raw actor output stays effectively in [-1, 1] (no separate `clip` needed).
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

        # Gaussian noise models the residual sensor/filter error.
        # >>> CHANGED: pitch noise 0.02 -> 0.01 rad. Near balance the true tilt is only ~0.03 rad, so
        # std=0.02 (~1.1 deg) buried the angle signal in noise and the policy ignored it. 0.01 keeps a
        # usable signal-to-noise ratio and matches a good complementary filter. Rate noise unchanged.
        pitch_angle = ObsTerm(func=mdp.imu_pitch_angle, noise=Gnoise(mean=0.0, std=0.01))
        pitch_rate = ObsTerm(func=mdp.imu_pitch_rate, noise=Gnoise(mean=0.0, std=0.02))
        # encoder reads both wheels' angular velocity (rad/s) -- disabled to retrain with input of pitch angle/rate only
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

    # wheel - floor friction: set once at startup (narrow range = fixed value, not randomized per episode),
    # matching the ground's physics_material (static=1.5, dynamic=1.2) so the wheels grip the floor and don't slip.
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

    # randomize the chassis center of mass (base_link, which merged upper_base/battery/bolts/motors via
    # merge_fixed_joints) so the policy doesn't overfit to one ideal CoM and is more robust when the real CoM is offset.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (-0.001, 0.001)},
        },
    )

    # the wheels' true axis = X_robot = "roll" in the standard roll/pitch/yaw convention (rotation about X/Y/Z),
    # so randomizing the initial tilt angle must use the key "roll", not "pitch"
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

    # random push along the Y axis (world), perpendicular to the motor rotation axis (X_robot),
    # every 2-4s, simulating disturbance/collision so the robot learns to recover balance via wheel torque
    # >>> CHANGED: softened the disturbance for stable early learning. Was y +-2.0 m/s every 1-3 s,
    # which can topple this small robot past the fall limit before it has learned anything. Now
    # +-1.0 m/s every 2-4 s. Once balancing is learned you can raise this back for more robustness.
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

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    # >>> CHANGED: -1000 -> -50. The old value was ~200x the per-step rewards (~+-5) and blew up the
    # PPO value function early on (the robot falls constantly at the start), a prime cause of "always
    # fails". -50 still discourages falling while letting the alive/upright signal drive learning.
    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # (3) Primary task: keep the chassis upright (projected_gravity_b deviates from [0,0,-1])
    upright = RewTerm(
        func=mdp.base_upright_penalty,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # (3b) Bonus: positive reward (exponential, upper-bounded at 1.0) when upright, complementing "upright" above
    upright_bonus = RewTerm(
        func=mdp.base_upright_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.2},
    )
    # (4) Shaping: reduce tilt angular rate (damp oscillation)
    pitch_rate = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # (5) Yaw-rate shaping.
    # >>> DISABLED: yaw is NOT in the 2-D observation, so the policy cannot control it -> the term only
    # adds reward variance. With the symmetric 1-D action both wheels get equal torque, so the robot
    # barely yaws anyway. Re-enable only if you later add wheel-encoder terms to the observation.
    # yaw_rate = RewTerm(
    #     func=mdp.ang_vel_z_l2,
    #     weight=-0.75,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # (6) Forward-drift shaping.
    # >>> DISABLED: (a) linear velocity is not observable by the 2-D [pitch, pitch_rate] policy, so it
    # cannot act on it; (b) it also penalised the WRONG axis -- the wheels spin about X_robot so the
    # cart rolls along Y_robot, not X. Like the real IMU-only robot, some slow drift is expected/OK.
    # lin_vel_x = RewTerm(
    #     func=mdp.lin_vel_x_l2,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    # (7) Shaping: penalize the two wheels spinning in opposite directions/at different speeds -- disabled together with removing encoder/wheel_vel
    # wheel_vel_diff = RewTerm(
    #     func=mdp.wheel_vel_diff_l2,
    #     weight=-0.0000,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )
    # (8) Shaping: smooth action, avoid motor jerk
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # (9) Shaping: reduce action magnitude, prefer near-zero action once balanced
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.000)
    # (10) Shaping: penalize wheel velocity (read from encoder) -- disabled together with removing encoder/wheel_vel
    # wheel_vel_l2 = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )
    # (11) PD-imitation shaping.
    # >>> DISABLED: we want a PURE RL policy, not a PID imitator (per the deployment goal). Its sign
    # was also explicitly unverified in rewards.py -- if wrong, it actively pulls the policy toward a
    # DE-stabilising controller and fights the upright reward (a top suspect for "always fails").
    # Only re-enable if pure RL cannot converge AND you have verified the kp/kd sign in the sim first.
    # pid_mimic = RewTerm(
    #     func=mdp.pid_mimic_l2,
    #     weight=-1.0,
    #     params={"kp": 8.0, "kd": 0.5, "angle_setpoint": 0.0},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fell past the allowed angle (~30°, computed from projected_gravity_b)
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
