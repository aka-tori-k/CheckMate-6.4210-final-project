import os
import numpy as np
# from pydrake.all import RigidTransform, RotationMatrix
from grasp_planning.mesh_grasp_sampling import (antipodal_pairs, score_grasp, gripper_pose_from_pair)
# from generate_grasps import (generate_mesh_grasps, get_mesh_path, transform_to_dict)


# def sample_surface_points_from_drake_cloud(cloud_np, n_samples=200):
#     # Use a smaller radius and fewer neighbors
#     cloud.EstimateNormals(radius=0.01, num_closest=10)

#     # Get numpy arrays
#     points = cloud.xyzs().T
#     normals = cloud.normals().T

#     # Remove NaN normals
#     mask = ~np.isnan(normals).any(axis=1)
#     points = points[mask]
#     normals = normals[mask]

#     # Downsample
#     N = points.shape[0]
#     idx = np.random.choice(N, min(N, n_samples), replace=False)

#     return points[idx], normals[idx]



def estimate_normals_pca(points, k=20):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, idx = nbrs.kneighbors(points)

    normals = []
    for i in range(len(points)):
        p_neighbors = points[idx[i]]
        cov = np.cov(p_neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]     # smallest eigenvalue
        normals.append(normal)

    return np.array(normals)

# def compute_normals_drake(cloud, radius=0.02, num_closest=30):
#     """
#     cloud: pydrake.perception.PointCloud
#     Modifies cloud in-place to add normals.
#     Returns normals as (N,3) numpy array.
#     """

#     # Preconditions
#     assert cloud.has_xyzs(), "PointCloud must have XYZ before estimating normals."

#     success = cloud.EstimateNormals(radius=radius, num_closest=num_closest)

#     if not success:
#         print("[WARN] Some normals could not be estimated (NaNs present).")

#     normals = cloud.normals().T  # Drake stores normals as (3, N)
#     return normals

def generate_grasp_from_point_cloud(point_cloud, n_candidates=8, n_samples=2000):
    """
    Returns a list of 8 RigidTransforms (object → gripper).
    """
    print("started grasp generation")
    N = point_cloud.shape[0]
    idx = np.random.choice(N, min(N, n_samples), replace=False)
    points = point_cloud[idx]

    # estimate normals via PCA
    normals = estimate_normals_pca(points)
    print("got normals and points")
    # Generate antipodal index pairs
    pairs = antipodal_pairs(points, normals) # returns tuples (i, j) if p_i, p_j are antipodal
    print("got pairs")

    if len(pairs) == 0:
        raise RuntimeError("No antipodal grasp candidates found. Increase samples or thresholds.")

    scored = []

    for (i, j) in pairs:
        p_i, p_j = points[i], points[j]
        n_i, n_j = normals[i], normals[j]

        score = score_grasp(p_i, p_j, n_i, n_j)
        pose = gripper_pose_from_pair(p_i, p_j, n_i, n_j)

        # scored.append((score, pose))
        scored.append({
            "score": score,
            "pose": pose,       # still a RigidTransform
            "p_i": p_i.tolist(),
            "p_j": p_j.tolist(),
            "n_i": n_i.tolist(),
            "n_j": n_j.tolist(),
        })

    # Sort by score descending
    # scored.sort(key=lambda x: -x[0])
    scored.sort(key=lambda x: -x["score"])
    top_n = scored[:n_candidates]
    print(top_n)

    # Return top N = 8
    # return [pose for score, pose in scored[:n_candidates]]
    return top_n
