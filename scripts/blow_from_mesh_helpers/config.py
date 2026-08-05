import os
import sys
from collections.abc import Mapping

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")
CONFIG_PATH_BY_ROBOT = {
    "doosan": os.path.join(CONFIG_DIR, "blow_from_mesh_doosan.yaml"),
    "ur": os.path.join(CONFIG_DIR, "blow_from_mesh_ur.yaml"),
}
DETECTOR_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "KIMM_chipblowing_detection")
DETECTOR_DIR_CANDIDATES = [
    CONFIGURED_DETECTOR_DIR
    for CONFIGURED_DETECTOR_DIR in (
        os.environ.get("KIMM_CHIPBLOWING_DETECTION_DIR"),
        DETECTOR_DIR,
        os.path.join("/ros2_ws", "src", "KIMM_chipblowing_detection"),
    )
    if CONFIGURED_DETECTOR_DIR
]
for detector_dir in DETECTOR_DIR_CANDIDATES:
    if detector_dir not in sys.path:
        sys.path.append(detector_dir)

QUERY_PIXELS = [
    # (640.0, 360.0),
]


def consume_cli_option(*names: str) -> str | None:
    """Read and remove script-only args before rclpy parses sys.argv."""
    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == "--ros-args":
            break
        if arg in names:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{arg} requires a value.")
            value = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            return value
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                value = arg[len(prefix):]
                del sys.argv[index]
                return value
        index += 1
    return None


def resolve_config_path() -> tuple[str, str | None]:
    explicit_config = (
        consume_cli_option("--config", "--config-file")
        or os.environ.get("BLOW_FROM_MESH_CONFIG")
    )
    configured_robot = (
        consume_cli_option("--robot", "--robot-type")
        or os.environ.get("BLOW_FROM_MESH_ROBOT")
    )

    if explicit_config:
        selected_robot = configured_robot.strip().lower() if configured_robot else None
        return explicit_config, selected_robot

    selected_robot = (configured_robot or "doosan").strip().lower()
    if selected_robot not in CONFIG_PATH_BY_ROBOT:
        raise ValueError(
            f"Unsupported robot selection '{selected_robot}'. Use doosan or ur."
        )
    return CONFIG_PATH_BY_ROBOT[selected_robot], selected_robot


CONFIG_PATH, SELECTED_ROBOT_TYPE = resolve_config_path()


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rotation_from_config(cfg: dict) -> R:
    if "matrix" in cfg:
        return R.from_matrix(np.array(cfg["matrix"], dtype=float))
    if "quaternion_xyzw" in cfg:
        return R.from_quat(np.array(cfg["quaternion_xyzw"], dtype=float))
    if "euler_xyz" in cfg:
        return R.from_euler("xyz", cfg["euler_xyz"], degrees=cfg.get("degrees", False))
    raise ValueError("Rotation config must contain 'matrix', 'quaternion_xyzw', or 'euler_xyz'.")


def translation_from_config(cfg: dict) -> np.ndarray:
    return np.array(cfg.get("translation", [0.0, 0.0, 0.0]), dtype=float)


def world_to_cnc_from_config(cfg: object) -> tuple[R, np.ndarray]:
    """Parse the world-from-CNC pose, including the legacy translation form."""
    if isinstance(cfg, Mapping):
        try:
            rotation = rotation_from_config(cfg)
            translation = translation_from_config(cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mesh.world_to_cnc pose: {exc}") from exc
    else:
        rotation = R.identity()
        try:
            translation = np.array(cfg, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "mesh.world_to_cnc must be a pose mapping or three-value "
                "legacy translation list."
            ) from exc

    if translation.shape != (3,):
        raise ValueError(
            "mesh.world_to_cnc translation must contain exactly three values."
        )
    return rotation, translation


CONFIG = load_config()

ROBOT_ID = CONFIG["robot"]["id"]
ROBOT_MODEL = CONFIG["robot"]["model"]
ROBOT_TYPE = CONFIG["robot"].get("type", "doosan")
if ROBOT_TYPE not in ("doosan", "ur"):
    raise ValueError(f"Unsupported robot.type '{ROBOT_TYPE}'. Use doosan or ur.")
if SELECTED_ROBOT_TYPE and ROBOT_TYPE != SELECTED_ROBOT_TYPE:
    raise ValueError(
        f"Selected robot '{SELECTED_ROBOT_TYPE}' but loaded config "
        f"robot.type is '{ROBOT_TYPE}' from {CONFIG_PATH}."
    )
print(f"[config] Loaded {ROBOT_TYPE} config: {CONFIG_PATH}")
TRANSFORMS = CONFIG["transforms"]
GAZ_TO_OPT_R = rotation_from_config(TRANSFORMS["gazebo_to_optical"])
GAZ_TO_OPT_T = translation_from_config(TRANSFORMS["gazebo_to_optical"])
WLD_TO_BASE_R = rotation_from_config(TRANSFORMS["world_to_base"])
WLD_TO_BASE_T = translation_from_config(TRANSFORMS["world_to_base"])
L6_TO_CAM_R = rotation_from_config(TRANSFORMS["link_to_camera"])
L6_TO_CAM_T = translation_from_config(TRANSFORMS["link_to_camera"])
WLD_TO_CNC_R, WLD_TO_CNC_T = world_to_cnc_from_config(
    CONFIG["mesh"].get("world_to_cnc", [0.0, 0.0, 0.0])
)
# Retain the translation-only name for external scripts using the old constant.
WLD_TO_CNC = WLD_TO_CNC_T
CNC_MESH_LOCAL_OFFSET = np.array(
    CONFIG["mesh"].get("mesh_local_offset_m", [0.0, 0.0, 0.0]), dtype=float
)
if CNC_MESH_LOCAL_OFFSET.shape != (3,):
    raise ValueError("mesh.mesh_local_offset_m must contain exactly three values.")

MESH_DIMENSION = float(CONFIG["mesh"]["dimension"])
CAMERA_CFG = CONFIG["camera"]
fx = float(CAMERA_CFG["intrinsics"]["fx"])
fy = float(CAMERA_CFG["intrinsics"].get("fy", fx))
cx_k = float(CAMERA_CFG["intrinsics"]["cx"])
cy_k = float(CAMERA_CFG["intrinsics"]["cy"])
camera_K = np.array([[fx, 0, cx_k], [0, fy, cy_k], [0, 0, 1]])

CNC_mesh_path = CONFIG["mesh"]["path"]
if not os.path.isabs(CNC_mesh_path):
    CNC_mesh_path = os.path.join(SCRIPT_DIR, CNC_mesh_path)

IMG_TOPIC = CAMERA_CFG["image_topic"]
INIT_TARGET = CONFIG["robot"].get("initial_target")
if INIT_TARGET is None and "init_posx" in CONFIG["robot"]:
    INIT_TARGET = {"type": "doosan_posx", "values": CONFIG["robot"]["init_posx"]}
if INIT_TARGET is None:
    raise ValueError("robot.initial_target is required.")
TF_SOURCE_FRAME = CONFIG["robot"]["frames"]["tf_source"]
TF_TARGET_FRAME = CONFIG["robot"]["frames"]["tf_target"]
MOVEIT_CFG = CONFIG["moveit"]
MOVEIT_GROUP = MOVEIT_CFG["group"]
if ROBOT_TYPE == "ur" and MOVEIT_GROUP == "manipulator":
    print(
        "[config] robot.type=ur is using moveit.group='manipulator'. "
        "Most UR MoveIt configs use 'ur_manipulator'; set moveit.group to "
        "the group name in your loaded UR SRDF/ompl_planning.yaml."
    )
MOVEIT_EE_LINK = MOVEIT_CFG["ee_link"]
MOVEIT_BASE_FRAME = MOVEIT_CFG["base_frame"]
MOVEIT_PLANNING_SERVICE = MOVEIT_CFG["planning_service"]
MOVEIT_MOVE_ACTION = MOVEIT_CFG["move_action"]
MOVEIT_SCENE_SERVICE = MOVEIT_CFG["scene_service"]
MOVEIT_SCENE_TOPIC = MOVEIT_CFG["scene_topic"]
MOVEIT_EXECUTE_ACTION = MOVEIT_CFG["execute_action"]
MOVEIT_PLANNING_TIME = float(MOVEIT_CFG["planning_time"])
MOVEIT_PLANNING_ATTEMPTS = int(MOVEIT_CFG["planning_attempts"])
MOVEIT_POS_TOLERANCE = float(MOVEIT_CFG["position_tolerance"])
MOVEIT_ORI_TOLERANCE = float(MOVEIT_CFG["orientation_tolerance"])
MOVEIT_VELOCITY_SCALING = float(MOVEIT_CFG.get("velocity_scaling", 0.2))
MOVEIT_ACCELERATION_SCALING = float(MOVEIT_CFG.get("acceleration_scaling", 0.2))
MOVEIT_JOINT_NAMES = MOVEIT_CFG.get(
    "joint_names",
    ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
)
MOVEIT_JOINT_TOLERANCE = float(MOVEIT_CFG.get("joint_tolerance", 0.001))
if ROBOT_TYPE == "ur" and MOVEIT_JOINT_TOLERANCE < 0.01:
    print(
        "[config] robot.type=ur needs a wider joint goal tolerance for OMPL "
        f"sampling; overriding moveit.joint_tolerance {MOVEIT_JOINT_TOLERANCE} -> 0.01."
    )
    MOVEIT_JOINT_TOLERANCE = 0.01
CNC_COLLISION_OBJECT_ID = MOVEIT_CFG["collision_object_id"]
ALLOW_CNC_COLLISION_LINKS = MOVEIT_CFG["allow_cnc_collision_links"]

UR_DEFAULT_DISABLED_SELF_COLLISION_PAIRS = [
    ("base_link_inertia", "robot_base_guard"),
    ("shoulder_link", "upper_arm_link"),
    ("upper_arm_link", "forearm_link"),
    ("forearm_link", "wrist_1_link"),
    ("wrist_1_link", "wrist_2_link"),
    ("wrist_2_link", "wrist_3_link"),
]

disabled_self_collision_pairs = [
    tuple(pair) for pair in MOVEIT_CFG["disabled_self_collision_pairs"]
]
if ROBOT_TYPE == "ur":
    disabled_self_collision_pairs.extend(UR_DEFAULT_DISABLED_SELF_COLLISION_PAIRS)
DISABLED_SELF_COLLISION_PAIRS = list(dict.fromkeys(disabled_self_collision_pairs))
DETECTION_STAGING_TARGET = CONFIG["robot"].get("detection_staging_target")
if DETECTION_STAGING_TARGET is None:
    DETECTION_STAGING_TARGET = {
        "type": "joint_values",
        "values": CONFIG["robot"].get("all_zero_joints", [0.0] * len(MOVEIT_JOINT_NAMES)),
    }
