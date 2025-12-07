import numpy as np
import trimesh
from pydrake.all import RigidTransform, RotationMatrix

# -------------------------
# Mesh -> Piece frame helpers
# -------------------------
def rotation_x_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]])

# In your SDF: <pose>0 0 0 1.5708 0 0</pose> --> rotate +90 deg about X
SDF_ROT_X90 = rotation_x_matrix(np.pi / 2.0)


# ---------------------------------------------------------
# Utility functions (fixed + improved)
# ---------------------------------------------------------
def sample_surface_points(mesh, n_samples=2000,
                          apply_sdf_rotation=True, mesh_scale=4.0):
    """
    Sample points on the raw OBJ mesh, then transform points+normals into
    the 'piece' frame (the SDF frame) by applying the SDF rotation and scale.

    - mesh: trimesh.Trimesh (loaded from OBJ)
    - apply_sdf_rotation: apply the 90deg X rotation present in your SDF.
    - mesh_scale: uniform scale listed in SDF (e.g. 4).
    Returns: points_piece (Nx3), normals_piece (Nx3) expressed in piece frame.
    """
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]

    # Apply uniform scale first (affects points, normals unaffected by uniform scale direction)
    if mesh_scale is not None:
        pts = pts * mesh_scale

    # Apply SDF rotation (maps from OBJ mesh frame -> piece frame)
    if apply_sdf_rotation:
        R = SDF_ROT_X90
        pts = (R @ pts.T).T
        normals = (R @ normals.T).T

    # Ensure normals are normalized (numerical safety)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    return pts, normals


def antipodal_pairs(points, normals, angle_thresh=np.deg2rad(20), dist_thresh=0.03):
    """
    Brute-force (simple) antipodal pair finder.
    Returns list of (i, j) indices where:
      - points are within dist_thresh
      - normals are roughly opposite (within angle_thresh)
    Note: O(N^2). For large N, replace with KD-tree neighbor search.
    """
    pairs = []
    N = len(points)
    cos_allowed = -np.cos(angle_thresh)  # dot <= this => sufficiently opposite

    for i in range(N):
        pi = points[i]
        ni = normals[i]
        for j in range(i + 1, N):
            pj = points[j]
            nj = normals[j]

            if np.linalg.norm(pi - pj) > dist_thresh:
                continue

            # normals should be near-opposite
            if np.dot(ni, nj) > cos_allowed:
                continue

            pairs.append((i, j))

    return pairs


def gripper_pose_from_pair(p_i, p_j, n_i, n_j):
    """
    Construct a RigidTransform (piece frame) for a parallel-jaw gripper:
      - x: finger closing axis (p_j - p_i)
      - z: approach direction (prefer -average_normal)
      - y: z x x
    Returns None if degenerate/unusable.
    """
    center = 0.5 * (p_i + p_j)

    # x axis: finger closing direction
    x = p_j - p_i
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x = x / nx

    # z axis: prefer - (n_i + n_j)
    z_raw = -(n_i + n_j)
    if np.linalg.norm(z_raw) < 1e-3:
        # fallback: use -n_i (one contact normal) as approach if sum is tiny
        z_raw = -n_i
    z = z_raw / (np.linalg.norm(z_raw) + 1e-12)

    # If z ended up nearly parallel to x, pick an alternative (avoid degeneracy)
    if abs(np.dot(x, z)) > 0.98:
        # choose a vector not parallel to x: try world z, then world x
        alt = np.array([0.0, 0.0, -1.0])  # prefer top-down approach as default
        if abs(np.dot(alt, x)) > 0.98:
            alt = np.array([1.0, 0.0, 0.0])
        z = alt - np.dot(alt, x) * x  # make it orthogonal to x
        if np.linalg.norm(z) < 1e-6:
            return None
        z /= np.linalg.norm(z)

    # y = z x x (right-handed)
    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y = y / ny

    # Re-orthonormalize x = y x z
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    R = RotationMatrix(np.column_stack((x, y, z)))
    return RigidTransform(R, center)


def score_grasp(p_i, p_j, n_i, n_j,
                prefer_top_down_weight=1.5,
                normal_align_weight=2.0,
                dist_weight=0.05):
    """
    Score higher = better.
    Includes:
      - normal alignment (antipodality)
      - distance between contacts (prefer wider within gripper limits)
      - penalize vertical mismatch between the two contact points
      - reward approach roughly top-down (z aligned with [0,0,-1])
      - penalize axis degeneracy
    """
    # closing direction
    x = (p_j - p_i)
    x /= (np.linalg.norm(x) + 1e-9)

    # approach vector
    z_raw = -(n_i + n_j)
    if np.linalg.norm(z_raw) < 1e-3:
        z_raw = -n_i
    z = z_raw / (np.linalg.norm(z_raw) + 1e-9)

    normal_alignment = -np.dot(n_i, n_j)  # close to +1 is good
    dist = np.linalg.norm(p_i - p_j)
    dz = abs(p_i[2] - p_j[2])
    dist_from_center = abs((p_i[2] + p_j[2]) * 0.5 - 0.06)  # center height bias

    axis_parallel = abs(np.dot(x, z))  # 0 good, 1 bad

    # reward approach that points downward (piece-frame negative z)
    preferred_approach = np.array([0.0, 0.0, -1.0])
    approach_alignment = np.dot(z, preferred_approach)  # 1 if perfectly top-down

    score = (
        normal_align_weight * normal_alignment +
        dist_weight * dist -
        0.3 * dz -
        0.01 * dist_from_center -
        0.5 * axis_parallel +
        prefer_top_down_weight * approach_alignment
    )
    return float(score)


# ---------------------------------------------------------
# Top-level API (fixed)
# ---------------------------------------------------------
def generate_mesh_grasps(mesh_path,
                         n_candidates=8,
                         n_samples=2000,
                         apply_sdf_rotation=True,
                         mesh_scale=4.0,
                         dist_thresh=0.03,
                         angle_thresh=np.deg2rad(20)):
    """
    Returns a list of up to n_candidates RigidTransforms in the *piece* frame
    (i.e., the frame after applying the SDF <pose> rotation/scale).
    """
    # Load raw OBJ with trimesh
    mesh = trimesh.load(mesh_path)

    # Sample and transform into piece frame
    points, normals = sample_surface_points(mesh, n_samples,
                                           apply_sdf_rotation=apply_sdf_rotation,
                                           mesh_scale=mesh_scale)

    pairs = antipodal_pairs(points, normals, angle_thresh=angle_thresh, dist_thresh=dist_thresh)
    if len(pairs) == 0:
        raise RuntimeError("No antipodal grasp candidates found. Increase samples or thresholds.")

    scored = []
    for (i, j) in pairs:
        p_i, p_j = points[i], points[j]
        n_i, n_j = normals[i], normals[j]

        pose = gripper_pose_from_pair(p_i, p_j, n_i, n_j)
        if pose is None:
            continue

        score = score_grasp(p_i, p_j, n_i, n_j)
        scored.append((score, pose))

    if len(scored) == 0:
        raise RuntimeError("No valid (non-degenerate) grasps after pose construction.")

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [pose for (score, pose) in scored[:n_candidates]]
    return top


# -------------------------
# Example usage
# -------------------------
# if __name__ == "__main__":
#     mesh_path = "/workspaces/CheckMate-6.4210-final-project/src/models/pieces/pawns/pawn_mesh.obj"
#     grasps = generate_mesh_grasps(mesh_path, n_candidates=8, n_samples=3000,
#                                   apply_sdf_rotation=True, mesh_scale=4.0)
#     print(f"Found {len(grasps)} grasp poses (RigidTransform) in piece frame.")
#     # Example: print first transform
#     for i, g in enumerate(grasps):
#         R = g.rotation().matrix()
#         t = g.translation()
#         print(f"Grasp {i}: R=\n{R}\n t={t}")
