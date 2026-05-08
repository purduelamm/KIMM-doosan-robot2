import rclpy

from .config import CONFIG, ROBOT_ID, ROBOT_MODEL, ROBOT_TYPE

_node = None
_air_node = None
_dsr = {}
ROBOT_MODE_AUTONOMOUS = None


class _NodeProxy:
    def __getattr__(self, name):
        if _node is None:
            raise RuntimeError("ROS node is not initialized. Call init_ros() first.")
        return getattr(_node, name)


class _AirNodeProxy:
    def __getattr__(self, name):
        if _air_node is None:
            raise RuntimeError("Tool control node is not initialized. Call init_ros() first.")
        return getattr(_air_node, name)


class _DSRFunctionProxy:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        try:
            fn = _dsr[self.name]
        except KeyError as exc:
            raise RuntimeError("DSR_ROBOT2 is not initialized. Call init_ros() first.") from exc
        return fn(*args, **kwargs)


node = _NodeProxy()
air_node = _AirNodeProxy()
movej = _DSRFunctionProxy("movej")
posj = _DSRFunctionProxy("posj")
movel = _DSRFunctionProxy("movel")
posx = _DSRFunctionProxy("posx")
set_robot_mode = _DSRFunctionProxy("set_robot_mode")
get_current_posx = _DSRFunctionProxy("get_current_posx")
get_current_tool_flange_posx = _DSRFunctionProxy("get_current_tool_flange_posx")


class DisabledToolControl:
    def tool_airgun(self, state: bool):
        print(f"[tool] Tool control disabled; ignoring airgun={state}.")


def is_doosan_robot() -> bool:
    return ROBOT_TYPE == "doosan"


def _create_tool_control():
    tool_cfg = CONFIG.get("tool_control", {})
    if not bool(tool_cfg.get("enabled", False)):
        return DisabledToolControl()

    tool_type = tool_cfg.get("type", "doosan_tool_digital_output")
    if tool_type == "doosan_tool_digital_output":
        if not is_doosan_robot():
            print("[tool] Doosan tool digital output is unavailable for non-Doosan robots.")
            return DisabledToolControl()
        from tool_control import ToolControlNode

        return ToolControlNode(_node)

    raise ValueError(f"Unsupported tool_control.type '{tool_type}'.")


def init_ros():
    global _node, _air_node, ROBOT_MODE_AUTONOMOUS

    if _node is not None:
        return _node

    if not rclpy.ok():
        rclpy.init()

    node_kwargs = {}
    if ROBOT_ID:
        node_kwargs["namespace"] = ROBOT_ID
    _node = rclpy.create_node(CONFIG["robot"].get("node_name", "coverage_path"), **node_kwargs)

    if is_doosan_robot():
        import DR_init

        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = _node

        from DSR_ROBOT2 import (
            ROBOT_MODE_AUTONOMOUS as _ROBOT_MODE_AUTONOMOUS,
            get_current_posx as _get_current_posx,
            get_current_tool_flange_posx as _get_current_tool_flange_posx,
            movej as _movej,
            movel as _movel,
            posj as _posj,
            posx as _posx,
            set_robot_mode as _set_robot_mode,
        )

        _dsr.update(
            {
                "movej": _movej,
                "posj": _posj,
                "movel": _movel,
                "posx": _posx,
                "set_robot_mode": _set_robot_mode,
                "get_current_posx": _get_current_posx,
                "get_current_tool_flange_posx": _get_current_tool_flange_posx,
            }
        )
        ROBOT_MODE_AUTONOMOUS = _ROBOT_MODE_AUTONOMOUS

    _air_node = _create_tool_control()
    return _node
