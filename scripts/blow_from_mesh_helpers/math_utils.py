import numpy as np


def make_SE3(R_mat: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def make_mesh_world_transform(
    R_w_cnc: np.ndarray,
    t_w_cnc_m: np.ndarray,
    mesh_local_offset_m: np.ndarray,
    mesh_units_per_metre: float,
) -> np.ndarray:
    """Return the native-mesh-to-world transform expressed in mesh units.

    The local offset locates the native mesh in the CNC frame, so it is
    rotated by the CNC orientation before the CNC's world translation is
    applied. Mesh vertices remain in their native units after transformation.
    """
    rotation = np.asarray(R_w_cnc, dtype=float)
    translation = np.asarray(t_w_cnc_m, dtype=float)
    local_offset = np.asarray(mesh_local_offset_m, dtype=float)
    units_per_metre = float(mesh_units_per_metre)

    if rotation.shape != (3, 3):
        raise ValueError("R_w_cnc must be a 3x3 rotation matrix.")
    if translation.shape != (3,):
        raise ValueError("t_w_cnc_m must contain exactly three values.")
    if local_offset.shape != (3,):
        raise ValueError("mesh_local_offset_m must contain exactly three values.")
    if not np.isfinite(units_per_metre) or units_per_metre <= 0.0:
        raise ValueError("mesh_units_per_metre must be a positive finite value.")

    mesh_translation = rotation @ local_offset + translation
    return make_SE3(rotation, mesh_translation * units_per_metre)
