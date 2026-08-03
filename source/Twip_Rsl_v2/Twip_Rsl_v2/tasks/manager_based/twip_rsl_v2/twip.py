# twip.py
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg


def _resolve_urdf(rel_path: str = "assets/TwoWheel.urdf") -> str:
    """Locate the URDF automatically, WITHOUT hardcoding the path (works on both Windows and Linux).

    Prefers the TWIP_URDF environment variable; otherwise walks UP from this file through the
    parent directories until it finds 'assets/TwoWheel.urdf' (repo root). Returns a posix path.
    """
    env_path = os.environ.get("TWIP_URDF")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve().as_posix()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / rel_path
        if candidate.is_file():
            return candidate.as_posix()
    raise FileNotFoundError(
        f"Khong tim thay '{rel_path}' khi di nguoc tu {here}. "
        "Dat bien moi truong TWIP_URDF tro toi file URDF neu de o noi khac."
    )


_URDF_PATH = _resolve_urdf()

##
# Configuration for the TWIP robot
##

TwoWheel_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # Spawn directly from the source URDF (no pre-built USD) to avoid config drift
        # whenever TwoWheel.urdf changes (link/inertial/collision).
    
        asset_path=_URDF_PATH,
        fix_base=False,
        root_link_name="base_link",
        # merge links joined by fixed joints (upper_base_link, battery, bolt1-4, motor1-2)
        # into base_link → leaving 3 rigid bodies: base_link, wheel1, wheel2.
        merge_fixed_joints=True,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            # don't bake the PD drive into the USD; the "wheels" actuator below sets gains at runtime
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link -> motor: z=-0.025; motor -> wheel: z=-0.0105; wheel radius=0.034
        # wheel bottom = -0.025 - 0.0105 - 0.034 = -0.0695 relative to base_link
        # → spawn base_link at z=0.0695 + margin (0.002) so the wheels just touch the ground
        pos=(0.0, 0.0, 0.0715),
        joint_pos={
            "wheel1_motor1_joint": 0.0,
            "wheel2_motor2_joint": 0.0,
        },
    ),
    actuators={
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["wheel1_motor1_joint", "wheel2_motor2_joint"],
            # real max torque of motor1/motor2
            effort_limit=0.49,
            stiffness=0.0,
            damping=0.002,
        ),
    },
)
