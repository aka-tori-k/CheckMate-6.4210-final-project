import numpy as np
import trimesh
from pydrake.all import RigidTransform, RotationMatrix

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def sample_surface_points(mesh, n_samples=2000):
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]
    return pts, normals


def antipodal_pairs(points, normals, angle_thresh=np.deg2rad(20), dist_thresh=0.03):
    """
    Returns index pairs (i,j) where points[i], points[j] form an antipodal pair:
      - Roughly opposite normals
      - Close enough to be gripped
    """
    pairs = []
    N = len(points)

    for i in range(N):
        for j in range(i+1, N):
            # Distances must be gripside small
            if np.linalg.norm(points[i] - points[j]) > dist_thresh:
                continue

            # Normals should face opposite directions
            if np.dot(normals[i], normals[j]) > -np.cos(angle_thresh):
                continue

            pairs.append((i, j))

    return pairs


def gripper_pose_from_pair(p_i, p_j, n_i, n_j):
    """
    Build a gripper frame in the *object coordinate frame*.

    Convention:
    - Gripper x-axis: pointing from left finger to right finger
    - Gripper z-axis: approach direction (approx. -normal)
    """

    # Center = midpoint between contact points
    center = 0.5 * (p_i + p_j)

    # Finger axis = direction from p_i to p_j
    x_axis = (p_j - p_i)
    x_axis /= np.linalg.norm(x_axis)

    # Approach direction = average inward normal
    z_axis = -(n_i + n_j)
    z_axis /= (np.linalg.norm(z_axis) + 1e-9)

    # y-axis = z × x  (right-handed)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-9)

    # Re-orthonormalize
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-9)

    R = RotationMatrix(np.column_stack((x_axis, y_axis, z_axis)))
    return RigidTransform(R, center)


def score_grasp(p_i, p_j, n_i, n_j):
    """
    Higher score = better grasp.
    Factors:
      - Parallelism of normals
      - Distance between contact points
    """
    normal_alignment = -np.dot(n_i, n_j)  # should be ~1
    dist = np.linalg.norm(p_i - p_j)
    return 1.0 * normal_alignment + 0.2 * dist


# ---------------------------------------------------------
# Top-level API
# ---------------------------------------------------------

def generate_mesh_grasps(mesh_path, n_candidates=8, n_samples=2000):
    """
    Returns a list of 8 RigidTransforms (object → gripper).
    """
    mesh = trimesh.load(mesh_path)
    points, normals = sample_surface_points(mesh, n_samples)

    # Generate antipodal index pairs
    pairs = antipodal_pairs(points, normals)

    if len(pairs) == 0:
        raise RuntimeError("No antipodal grasp candidates found. Increase samples or thresholds.")

    scored = []

    for (i, j) in pairs:
        p_i, p_j = points[i], points[j]
        n_i, n_j = normals[i], normals[j]

        score = score_grasp(p_i, p_j, n_i, n_j)
        pose = gripper_pose_from_pair(p_i, p_j, n_i, n_j)

        scored.append((score, pose))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    # Return top N = 8
    return [pose for score, pose in scored[:n_candidates]]
