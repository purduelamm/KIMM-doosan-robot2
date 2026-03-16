"""
Sample unique surface normals from a mesh given a top-view (x, y) coordinate and ROI radius R.

Pipeline:
  1. Load STL mesh and fix outward normals.
  2. Convert mesh to dense point cloud (sample points on each face, inherit face normal).
  3. Given top-view (x, y) + ROI radius R:
     a. Cast a downward ray to find the topmost visible surface z.
     b. Use (x, y, z) as the ROI centre.
     c. Collect all point cloud points within 3D radius R.
     d. Deduplicate their normals by angular threshold.
"""

import numpy as np
import trimesh


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    Load an STL mesh, zero its origin to the min bounding box corner,
    and fix face winding so every normal points outward.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a single mesh from {mesh_path}")

    # Zero the mesh: translate so that the min bounding box corner is at origin
    mesh.vertices -= mesh.bounds[0]

    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    if not mesh.is_winding_consistent:
        print("[load_mesh] WARNING: winding still inconsistent after repair. "
              "Some normals may point inward.")
    else:
        print(f"[load_mesh] Normals OK — winding consistent, {len(mesh.faces)} faces.")

    _ = mesh.face_normals
    return mesh


# ── Mesh → point cloud ────────────────────────────────────────────────────────

def mesh_to_pointcloud(
    mesh: trimesh.Trimesh,
    points_per_unit_area: float = None,
    total_points: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the mesh surface into a point cloud where every point carries the
    outward face normal of the triangle it was sampled from.

    Parameters
    ----------
    mesh                 : trimesh.Trimesh (normals already fixed outward)
    points_per_unit_area : if set, sample density per unit area of mesh surface
    total_points         : fallback total sample count when points_per_unit_area
                           is not set (default 50 000)

    Returns
    -------
    points  : (N, 3) sampled surface points
    normals : (N, 3) outward face normals at each point
    """
    if points_per_unit_area is not None:
        total_points = max(1000, int(mesh.area * points_per_unit_area))

    # trimesh.sample.sample_surface returns (points, face_indices)
    points, face_ids = trimesh.sample.sample_surface(mesh, total_points)
    normals = mesh.face_normals[face_ids]   # inherit face normal

    print(f"[mesh_to_pointcloud] Sampled {len(points):,} points "
          f"from {len(mesh.faces):,} faces  (area={mesh.area:.2f})")
    return points.astype(np.float64), normals.astype(np.float64)


# ── Surface z via ray ─────────────────────────────────────────────────────────

def find_topmost_z(
    mesh: trimesh.Trimesh,
    x: float,
    y: float,
) -> float | None:
    """
    Cast a ray downward from above (x, y) and return the z of the topmost
    upward-facing surface hit. Returns None if no hit.
    """
    z_start = mesh.bounds[1][2] + 1.0
    locations, _, face_ids = mesh.ray.intersects_location(
        ray_origins=np.array([[x, y, z_start]]),
        ray_directions=np.array([[0.0, 0.0, -1.0]]),
        multiple_hits=True,
    )
    if len(locations) == 0:
        return None

    hit_normals = mesh.face_normals[face_ids]
    upward = hit_normals[:, 2] > 0
    if upward.any():
        return float(locations[upward, 2].max())
    return float(locations[:, 2].max())   # fallback: any highest hit


# ── ROI query on point cloud ──────────────────────────────────────────────────

def sample_normals_in_roi(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    normals: np.ndarray,
    x: float,
    y: float,
    R: float,
    angular_threshold_deg: float = 5.0,
    z = None
) -> dict:
    """
    Find unique surface normals within ROI radius R of the surface point
    at top-view coordinate (x, y).

    Parameters
    ----------
    mesh                  : trimesh.Trimesh  (used only for ray cast)
    points                : (N, 3) point cloud positions
    normals               : (N, 3) point cloud normals
    x, y                  : top-view query coordinate (in mesh coordinate)
    R                     : ROI radius in mesh units
    angular_threshold_deg : angular deduplication threshold

    Returns
    -------
    dict:
        'surface_point'  : (3,) ROI centre on the mesh surface
        'roi_points'     : (K, 3) point cloud points inside ROI
        'all_normals'    : (K, 3) their normals
        'unique_normals' : (M, 3) deduplicated unique normals
    """
    # ── Step 1: find topmost surface z ───────────────────────────────────────
    if z == None:
        z_surface = find_topmost_z(mesh, x, y)
    else:
        print("got input z:", z)
        z_surface = z

    if z_surface is None:
        # No ray hit — find the closest point cloud point in XY and use its z
        xy_dist = np.linalg.norm(points[:, :2] - np.array([x, y]), axis=1)
        z_surface = float(points[np.argmin(xy_dist), 2])
        print(f"[sample_normals] No ray hit at ({x:.3f}, {y:.3f}). "
              f"Using nearest point z={z_surface:.3f}.")

    surface_point = np.array([x, y, z_surface])

    # ── Step 2: collect point cloud points within 3D sphere of radius R ──────
    dist_3d = np.linalg.norm(points - surface_point, axis=1)
    mask = dist_3d <= R
    roi_points  = points[mask]
    roi_normals = normals[mask]

    if len(roi_points) == 0:
        # Graceful fallback: return the single closest point
        closest_idx = np.argmin(dist_3d)
        roi_points  = points[[closest_idx]]
        roi_normals = normals[[closest_idx]]
        print(f"[sample_normals] No points within R={R}. "
              f"Returning closest point at distance {dist_3d[closest_idx]:.3f}.")

    # ── Step 3: cluster ROI points, keep only the closest cluster ───────────
    roi_points, roi_normals = _keep_closest_cluster(roi_points, roi_normals, surface_point)

    # ── Step 4: deduplicate normals ───────────────────────────────────────────
    unique_normals = _deduplicate_normals(roi_normals, angular_threshold_deg)

    return {
        "surface_point":  surface_point,
        "roi_points":     roi_points,
        "all_normals":    roi_normals,
        "unique_normals": unique_normals,
    }


# ── Spatial clustering ───────────────────────────────────────────────────────

def _keep_closest_cluster(
    points: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    eps_frac: float = 0.3,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the ROI points spatially with DBSCAN. If more than one cluster is
    found, keep only the one whose points are closest (minimum distance) to
    `center`. Noise points (label=-1) are discarded unless everything is noise,
    in which case all points are returned unchanged.

    Parameters
    ----------
    points      : (K, 3) ROI point positions
    normals     : (K, 3) corresponding normals
    center      : (3,)  ROI centre (surface_point)
    eps_frac    : DBSCAN eps as a fraction of the point cloud bounding-box
                  diagonal (auto-scaled so it works across mesh scales)
    min_samples : DBSCAN min_samples

    Returns
    -------
    filtered points  : (K', 3)
    filtered normals : (K', 3)
    """
    from sklearn.cluster import DBSCAN

    if len(points) < min_samples:
        return points, normals      # too few points to cluster

    # Auto-scale eps from the bounding box of the ROI point cloud
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    eps  = max(diag * eps_frac, 1e-6)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique_labels = set(labels) - {-1}   # exclude noise

    if len(unique_labels) <= 1:
        # 0 or 1 real cluster — nothing to filter
        if len(unique_labels) == 0:
            # everything is noise; return all as-is
            return points, normals
        # Single cluster: strip noise points
        mask = labels != -1
        return points[mask], normals[mask]

    # Multiple clusters — keep the one with minimum distance to center
    print(f"[cluster] {len(unique_labels)} clusters found in ROI — "
          f"keeping the one closest to surface point.")
    best_label   = None
    best_min_dist = np.inf
    for lbl in unique_labels:
        cluster_pts = points[labels == lbl]
        min_dist    = float(np.linalg.norm(cluster_pts - center, axis=1).min())
        if min_dist < best_min_dist:
            best_min_dist = min_dist
            best_label    = lbl

    mask = labels == best_label
    print(f"[cluster] Kept cluster {best_label}: "
          f"{mask.sum()} / {len(points)} points  "
          f"(min_dist={best_min_dist:.3f})")
    return points[mask], normals[mask]


# ── Normal deduplication ──────────────────────────────────────────────────────

def _deduplicate_normals(normals: np.ndarray, angular_threshold_deg: float) -> np.ndarray:
    """
    Greedy deduplication: keep a normal only if it differs by more than
    `angular_threshold_deg` from all already-kept normals.
    Anti-parallel normals (e.g. +Z / -Z) are treated as distinct.
    Normals with z < 0 (downward-facing) are ignored.
    """
    threshold_rad = np.deg2rad(angular_threshold_deg)
    norms = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    # Filter out downward-facing normals (z < 0)
    upward_mask = norms[:, 2] >= 0
    n_ignored = (~upward_mask).sum()
    if n_ignored > 0:
        print(f"[dedup] Ignored {n_ignored} downward-facing normals (z < 0).")
    norms = norms[upward_mask]

    unique = []
    for n in norms:
        if not unique:
            unique.append(n)
            continue
        dots   = np.clip(np.array(unique) @ n, -1.0, 1.0)
        angles = np.arccos(dots)          # NOT abs — keep anti-parallel as distinct
        if np.all(angles > threshold_rad):
            unique.append(n)

    return np.array(unique) if unique else np.empty((0, 3))


# ── Convenience wrapper ───────────────────────────────────────────────────────

def query_mesh_normals(
    # mesh_path: str,
    mesh: str,
    x: float,   # in mesh coordinate
    y: float,   # in mesh coordinate
    R: float,
    angular_threshold_deg: float = 5.0,
    total_points: int = 50_000_000,
    verbose: bool = True,
    z = None
) -> dict:
    """Full pipeline: load → point cloud → query normals."""
    # mesh    = load_mesh(mesh_path)
    pts, nrm = mesh_to_pointcloud(mesh, total_points=total_points)
    result  = sample_normals_in_roi(mesh, pts, nrm, x, y, R, angular_threshold_deg, z)

    if verbose:
        sp = result["surface_point"]
        print(f"Surface point      : ({sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f})")
        print(f"Points in ROI      : {len(result['roi_points'])}")
        print(f"Unique normals     : {len(result['unique_normals'])}  "
              f"(threshold={angular_threshold_deg}°)")
        for i, n in enumerate(result["unique_normals"]):
            print(f"  n[{i:02d}] = ({n[0]:+.6f},  {n[1]:+.6f},  {n[2]:+.6f})")

    return result



# ── SE(3) pose computation ────────────────────────────────────────────────────

def compute_se3_pose(result: dict, d: float) -> dict:
    """
    Compute an SE(3) pose from the unique normals and ROI surface point.

    Steps
    -----
    1. Sum all unique normals → v (then normalise).
    2. Position = surface_point + v * d   (d units away along v from ROI centre).
    3. Build rotation matrix R = [x_axis | y_axis | z_axis]:
         x_axis = -v                              (approach direction)
         y_axis = normalise(x_axis × world_Z)     (horizontal, perp to approach)
                  (if x_axis ∥ world_Z, fall back to world_X as reference)
         z_axis = normalise(y_axis × x_axis)      (guarantees world_Z · z_axis >= 0
                  because z_axis is constructed from the cross product that keeps
                  the frame right-handed and upward)
         Flip z_axis if world_Z · z_axis < 0.

    Parameters
    ----------
    result : dict returned by sample_normals_in_roi
    d      : standoff distance along v from the surface point

    Returns
    -------
    dict:
        'position'   : (3,)   position in world frame
        'rotation'   : (3, 3) rotation matrix  [x_col | y_col | z_col]
        'v'          : (3,)   normalised summed normal direction
        'T'          : (4, 4) homogeneous SE(3) transform
    """
    unique = result["unique_normals"]          # (M, 3), already unit vectors
    sp     = result["surface_point"]           # (3,)

    # ── Step 1: sum unique normals → v ───────────────────────────────────────
    v_raw = unique.sum(axis=0)
    v_norm = np.linalg.norm(v_raw)
    if v_norm < 1e-12:
        raise ValueError("Sum of unique normals is zero — cannot determine approach direction.")
    v = v_raw / v_norm                         # unit approach direction

    # ── Step 2: position ─────────────────────────────────────────────────────
    position = sp + v * d

    # ── Step 3: rotation matrix ───────────────────────────────────────────────
    world_Z = np.array([0.0, 0.0, 1.0])

    x_axis = -v                                # robot approaches along -v

    # y_axis must satisfy: world_Z · y_axis = 0  (horizontal)
    cross = np.cross(x_axis, world_Z)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-6:
        # x_axis is (anti-)parallel to world_Z — use world_X as fallback
        cross = np.cross(x_axis, np.array([1.0, 0.0, 0.0]))
        cross_norm = np.linalg.norm(cross)
    y_axis = cross / cross_norm

    # z_axis: right-hand rule, then ensure world_Z · z_axis >= 0
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    if np.dot(world_Z, z_axis) < 0:
        z_axis = -z_axis
        y_axis = np.cross(z_axis, x_axis)      # keep frame right-handed
        y_axis /= np.linalg.norm(y_axis)

    # Columns = axes expressed in world frame
    R_mat = np.column_stack([x_axis, y_axis, z_axis])  # (3, 3)

    # Homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3,  3] = position

    return {
        "position": position,
        "rotation": R_mat,
        "v":        v,
        "T":        T,
    }


def print_se3(pose: dict) -> None:
    """Pretty-print the SE(3) result."""
    p = pose["position"]
    R = pose["rotation"]
    v = pose["v"]
    T = pose["T"]
    print("\n── SE(3) Pose ──────────────────────────────────────")
    print(f"  Approach direction v : ({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f})")
    print(f"  Position             : ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})")
    print(f"  Rotation matrix (cols = x, y, z axes):")
    for row in R:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print(f"  Homogeneous T (4×4):")
    for row in T:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}  {row[3]:+.6f}]")


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_normals(result: dict, out_prefix: str = "normals"):
    """Save unique normals and surface point to .npy and .csv."""
    unique = result["unique_normals"]
    sp     = result["surface_point"]

    npy_path = f"{out_prefix}_unique.npy"
    csv_path = f"{out_prefix}_unique.csv"
    sp_path  = f"{out_prefix}_surface_pt.npy"

    np.save(npy_path, unique)
    np.save(sp_path,  sp)
    header = f"surface_point: {sp.tolist()}\nnx,ny,nz"
    np.savetxt(csv_path, unique, delimiter=",", header=header, comments="# ", fmt="%.8f")

    print(f"Saved {len(unique)} unique normals → {npy_path}  |  {csv_path}")
    print(f"Saved surface point              → {sp_path}")
    return {"npy": npy_path, "csv": csv_path, "surface_pt_npy": sp_path}


# ── Open3D visualization ──────────────────────────────────────────────────────

def _make_arrow(o3d, origin: np.ndarray, direction: np.ndarray,
                length: float, color: list, shaft_radius_frac: float = 0.06,
                cone_frac: float = 0.25) -> "o3d.geometry.TriangleMesh":
    """Build a solid arrow mesh from origin along direction."""
    shaft_r = length * shaft_radius_frac
    cone_h  = length * cone_frac
    shaft_h = length - cone_h

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_r,
        cone_radius=shaft_r * 2.0,
        cylinder_height=shaft_h,
        cone_height=cone_h,
    )
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()

    # Default arrow points in +Z; rotate to align with direction
    z = np.array([0.0, 0.0, 1.0])
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    axis  = np.cross(z, d)
    s     = np.linalg.norm(axis)
    c     = float(np.dot(z, d))
    if s < 1e-9:
        R_arr = np.eye(3) if c > 0 else np.diag([1, -1, -1]).astype(float)
    else:
        axis /= s
        K = np.array([[    0, -axis[2],  axis[1]],
                      [ axis[2],     0, -axis[0]],
                      [-axis[1],  axis[0],     0]])
        R_arr = np.eye(3) + s * K + (1 - c) * K @ K

    T = np.eye(4)
    T[:3, :3] = R_arr
    T[:3,  3] = origin
    arrow.transform(T)
    return arrow


def visualize_normals(mesh: trimesh.Trimesh, result: dict, normal_length: float = None):
    """
    Visualize in Open3D:
      • Mesh (grey)
      • ROI point cloud (blue)
      • Surface anchor sphere (red)
      • Unique normal arrows (coloured)
      • SE(3) pose frame as solid RGB arrows (X=red, Y=green, Z=blue)
        + white position sphere  — only when result["pose"] is present

    Parameters
    ----------
    mesh          : trimesh.Trimesh already loaded (no re-reading from disk)
    result        : dict returned by sample_normals_in_roi, optionally with "pose"
    normal_length : arrow length in mesh units (auto-scaled if None)
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Run: pip install open3d")
        return

    # Convert trimesh → Open3D mesh directly (no file I/O)
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.75, 0.75, 0.75])

    sp  = result["surface_point"]
    print("[VIZ]: sp ", sp)
    pts = result["roi_points"]

    # ── Auto-scale based on mesh diagonal (robust, not ROI-dependent) ─────────
    if normal_length is None:
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        normal_length = diag * 0.08   # 8% of mesh diagonal

    sphere_r = normal_length * 0.12

    geometries = [mesh_o3d]

    # ROI point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.2, 0.4, 1.0])
    geometries.append(pcd)

    # Surface anchor sphere
    anchor_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
    anchor_sphere.translate(sp)
    anchor_sphere.paint_uniform_color([1.0, 0.2, 0.2])
    anchor_sphere.compute_vertex_normals()
    geometries.append(anchor_sphere)

    # Unique normal arrows from surface point
    normal_colors = [[0.0, 0.8, 0.2], [1.0, 0.5, 0.0], [0.8, 0.0, 0.8],
                     [1.0, 1.0, 0.0], [0.0, 0.9, 0.9]]
    for i, n in enumerate(result["unique_normals"]):
        arrow = _make_arrow(o3d, sp, n, normal_length,
                            normal_colors[i % len(normal_colors)])
        geometries.append(arrow)

    # ── SE(3) pose frame ───────────────────────────────────────────────────────
    if "pose" in result:
        pose  = result["pose"]
        p     = pose["position"]
        R_m   = pose["rotation"]

        # Pose frame is drawn larger (1.5×) so it's distinct from surface normals
        frame_len = normal_length * 1.5

        axis_colors = [[1.0, 0.0, 0.0],   # X — red
                       [0.0, 0.9, 0.0],   # Y — green
                       [0.0, 0.3, 1.0]]   # Z — blue

        axis_labels = ["X", "Y", "Z"]
        for i in range(3):
            arrow = _make_arrow(o3d, p, R_m[:, i], frame_len, axis_colors[i])
            geometries.append(arrow)

        # Position sphere (white)
        pos_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r * 1.3)
        pos_sphere.translate(p)
        pos_sphere.paint_uniform_color([1.0, 1.0, 1.0])
        pos_sphere.compute_vertex_normals()
        geometries.append(pos_sphere)

        print(f"[visualize] Pose frame drawn at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
              f"arrow_length={frame_len:.3f}")
    
    # visualize coordinate frame
    pos_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r * 1.3)
    pos_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    pos_sphere.compute_vertex_normals()
    geometries.append(pos_sphere)
    frame_len = normal_length * 1.5

    axis_colors = [[1.0, 0.0, 0.0],   # X — red
                   [0.0, 0.9, 0.0],   # Y — green
                   [0.0, 0.3, 1.0]]   # Z — blue

    axis_labels = ["X", "Y", "Z"]
    R_axis = np.eye(3)
    for i in range(3):
        arrow = _make_arrow(o3d, [0,0,0], R_axis[:, i], frame_len, axis_colors[i])
        geometries.append(arrow)




    o3d.visualization.draw_geometries(
        geometries,
        window_name="Surface Normals + SE(3) Pose",
        width=1200, height=900,
    )


# ── Interactive top-view picker ───────────────────────────────────────────────

def pick_xy_from_topview(mesh: trimesh.Trimesh, R: float) -> tuple[float, float]:
    """
    Show a matplotlib top-view of the mesh. Click to choose (x, y).
    Press Enter or click Confirm to proceed.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.widgets import Button
    from matplotlib.collections import PolyCollection

    verts = mesh.vertices
    faces = mesh.faces
    xmin, ymin = verts[:, 0].min(), verts[:, 1].min()
    xmax, ymax = verts[:, 0].max(), verts[:, 1].max()
    margin = max(xmax - xmin, ymax - ymin) * 0.05

    print(f"[picker] Mesh XY bounds: x=[{xmin:.3f}, {xmax:.3f}]  "
          f"y=[{ymin:.3f}, {ymax:.3f}]")

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_axes([0.10, 0.10, 0.85, 0.85])
    fig.canvas.manager.set_window_title(
        "Top-view — click to select (x, y), then press Enter"
    )

    max_faces_draw = 20_000
    draw_faces = faces if len(faces) <= max_faces_draw else faces[
        np.random.choice(len(faces), max_faces_draw, replace=False)
    ]
    polys = verts[draw_faces][:, :, :2]
    pc = PolyCollection(polys, facecolor="none", edgecolor="#888888",
                        linewidth=0.3, alpha=0.6)
    ax.add_collection(pc)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (mesh units)")
    ax.set_ylabel("Y (mesh units)")
    ax.set_title("Click to set query point  |  Press Enter or click Confirm",
                 fontsize=11)

    state = {"x": None, "y": None}
    markers = []

    def _clear():
        for a in markers:
            try: a.remove()
            except Exception: pass
        markers.clear()

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        _clear()
        cx, cy = float(event.xdata), float(event.ydata)
        state["x"], state["y"] = cx, cy
        markers.extend([
            ax.axhline(cy, color="red", lw=0.8, ls="--", alpha=0.7),
            ax.axvline(cx, color="red", lw=0.8, ls="--", alpha=0.7),
            ax.plot(cx, cy, "r+", ms=14, mew=2)[0],
            ax.add_patch(mpatches.Circle((cx, cy), R, color="red",
                                         fill=False, lw=1.5)),
            ax.text(cx, cy, f"  ({cx:.2f}, {cy:.2f})", color="red",
                    fontsize=9, va="bottom"),
        ])
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "enter" and state["x"] is not None:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event",    _on_key)

    ax_btn = fig.add_axes([0.35, 0.01, 0.30, 0.05])
    btn = Button(ax_btn, "Confirm  (or press Enter)",
                 color="#d0e8ff", hovercolor="#90c8ff")
    btn.on_clicked(lambda _: plt.close(fig) if state["x"] is not None else None)

    plt.show()

    if state["x"] is None:
        raise RuntimeError("No point selected — click on the mesh in the top-view window.")

    print(f"[picker] Selected: x={state['x']:.6f},  y={state['y']:.6f}")
    return state["x"], state["y"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Sample unique surface normals from an STL mesh within a top-view ROI.\n"
            "If --x / --y are omitted, an interactive top-view window opens."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mesh", type=str,   help="Path to STL file")
    parser.add_argument("R",    type=float, help="ROI radius (mesh units)")
    parser.add_argument("--x",  type=float, default=None,
                        help="Top-view X (omit to pick interactively)")
    parser.add_argument("--y",  type=float, default=None,
                        help="Top-view Y (omit to pick interactively)")
    parser.add_argument("--threshold",   type=float, default=5.0,
                        help="Angular dedup threshold in degrees (default: 5.0)")
    parser.add_argument("--total-points", type=int, default=50_000,
                        help="Point cloud sample count (default: 50000)")
    parser.add_argument("--save-prefix", type=str, default="normals",
                        help="Output file prefix (default: 'normals')")
    parser.add_argument("--d",           type=float, default=None,
                        help="Standoff distance along summed normal to compute SE(3) pose")
    parser.add_argument("--visualize",   action="store_true",
                        help="Visualize with Open3D after processing")
    args = parser.parse_args()

    # Load mesh + build point cloud once
    mesh = load_mesh(args.mesh)
    pts, nrm = mesh_to_pointcloud(mesh, total_points=args.total_points)

    # Get (x, y)
    if args.x is None or args.y is None:
        print("No x/y provided — opening interactive top-view picker …")
        qx, qy = pick_xy_from_topview(mesh, args.R)
    else:
        qx, qy = args.x, args.y

    # Query
    result = sample_normals_in_roi(
        mesh, pts, nrm, qx, qy, args.R, args.threshold
    )

    sp = result["surface_point"]
    print(f"\nSurface point      : ({sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f})")
    print(f"Points in ROI      : {len(result['roi_points'])}")
    print(f"Unique normals     : {len(result['unique_normals'])}  "
          f"(threshold={args.threshold}°)")
    print("\n── Unique surface normals ──")
    for i, n in enumerate(result["unique_normals"]):
        print(f"  n[{i:02d}] = ({n[0]:+.6f},  {n[1]:+.6f},  {n[2]:+.6f})")

    save_normals(result, out_prefix=args.save_prefix)

    if args.d is not None:
        pose = compute_se3_pose(result, args.d)
        print_se3(pose)
        result["pose"] = pose   # attach so visualizer can draw it

    if args.visualize:
        visualize_normals(mesh, result)