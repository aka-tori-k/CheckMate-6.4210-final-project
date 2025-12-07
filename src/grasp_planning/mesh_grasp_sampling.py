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


# def gripper_pose_from_pair(p_i, p_j, n_i, n_j):
#     """
#     Build a gripper frame in the *object coordinate frame*.

#     Convention:
#     - Gripper x-axis: pointing from left finger to right finger
#     - Gripper z-axis: approach direction (approx. -normal)
#     """

#     # Center = midpoint between contact points
#     center = 0.5 * (p_i + p_j)
#     center += np.array([0,0,0.08])

#     # Finger axis = direction from p_i to p_j
#     x_axis = (p_j - p_i)
#     x_axis /= np.linalg.norm(x_axis)

#     # Approach direction = average inward normal
#     z_axis = -(n_i + n_j)
#     z_axis /= (np.linalg.norm(z_axis) + 1e-9)

#     # y-axis = z × x  (right-handed)
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis /= (np.linalg.norm(y_axis) + 1e-9)

#     # Re-orthonormalize
#     x_axis = np.cross(y_axis, z_axis)
#     x_axis /= (np.linalg.norm(x_axis) + 1e-9)

#     R = RotationMatrix(np.column_stack((x_axis, y_axis, z_axis)))
#     return RigidTransform(R, center)

def gripper_pose_from_pair(p_i, p_j, n_i, n_j):
    center = 0.5 * (p_i + p_j) # + np.array([0,0,0])

    # Try to build nominal axes
    x = p_j - p_i
    if np.linalg.norm(x) < 1e-6:
        return None  # reject grasp entirely
    x /= np.linalg.norm(x)

    z = -(n_i + n_j)
    if np.linalg.norm(z) < 1e-6:
        # fallback: use a perpendicular direction to x
        z = np.array([0,0,1])
        if abs(np.dot(z, x)) > 0.9:
            z = np.array([1,0,0])
    # print(type(z), type(np.linalg.norm(z)))
    z = z / np.linalg.norm(z)

    # Build y = z × x
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-6:
        # If still degenerate, pick arbitrary orthogonal
        y = np.cross([1,0,0], x)
        if np.linalg.norm(y) < 1e-6:
            y = np.cross([0,1,0], x)
    y /= np.linalg.norm(y)

    # Recompute x = y × z for orthonormality
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    R = RotationMatrix(np.column_stack((x, y, z)))
    return RigidTransform(R, center)


# def score_grasp(p_i, p_j, n_i, n_j):
#     """
#     Higher score = better grasp.
#     Factors:
#       - Parallelism of normals
#       - Distance between contact points
#     """
#     normal_alignment = -np.dot(n_i, n_j)  # should be ~1
#     dist = np.linalg.norm(p_i - p_j)
#     dz = abs(p_i[2] - p_j[2])
#     dist_from_center = abs(p_i[2] - 0.06)
#     return 2.0 * normal_alignment + 0.05 * dist - 0.01 * dist_from_center - 0.3 * dz
#     # points for antipodal and spaced out, penalized for change in z 
def score_grasp(p_i, p_j, n_i, n_j):
    x = (p_j - p_i)
    x /= (np.linalg.norm(x) + 1e-9)

    z = -(n_i + n_j)
    z /= (np.linalg.norm(z) + 1e-9)

    normal_alignment = -np.dot(n_i, n_j)
    dist = np.linalg.norm(p_i - p_j)
    dz = abs(p_i[2] - p_j[2])
    dist_from_center = abs(p_i[2] - 0.06)

    # New: penalize frame degeneracy
    axis_parallel = abs(np.dot(x, z))  # 0 good, 1 bad

    # positive => z points upward, which we do NOT want
    coming_from_below = max(0.0, np.dot(z, np.array([0,0,1])))

    return (
        2.0 * normal_alignment +
        0.05 * dist -
        0.01 * dist_from_center -
        0.3 * dz -
        0.5 * axis_parallel -
        2.0 * coming_from_below
    )


# ---------------------------------------------------------
# Top-level API
# ---------------------------------------------------------

def generate_mesh_grasps(mesh_path, n_candidates=8, n_samples=2000):
    """
    Returns a list of 8 RigidTransforms (object → gripper).
    """
    mesh = trimesh.load(mesh_path)
    # # Move mesh so its centroid is at the origin
    # mesh.apply_translation(-mesh.centroid)

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
    scored.sort(key=lambda x: x[0], reverse=True)
    # scored.sort(key=lambda x: -x["score"])

    # Return top N = 8
    return [pose for score, pose in scored[:n_candidates]]
    # return scored[:n_candidates]
