import numpy as np

def shortcut_path(path, collision_checker, num_iterations=100):
    """
    Shortcut a joint-space path to reduce unnecessary waypoints.

    Args:
        path: np.ndarray of shape (N, nq), joint-space path
        collision_checker: function q -> bool
            Returns True if joint position q is collision-free
        num_iterations: number of random shortcut attempts

    Returns:
        np.ndarray: shortcut path
    """
    
    path = np.array(path)
    N, nq = path.shape
    if N < 3:
        return path  # nothing to shortcut

    for _ in range(num_iterations):
        if len(path) < 3:
            break  # cannot shortcut further

        # randomly pick two indices along the path
        i, j = np.sort(np.random.choice(len(path), 2, replace=False))
        if j - i <= 1:
            continue  # adjacent points, nothing to shortcut

        q_start = path[i]
        q_end = path[j]

        # Check if linear interpolation between q_start and q_end is collision-free
        num_check = j - i + 1  # check each intermediate point
        ts = np.linspace(0, 1, num_check)
        collision_free = True
        for s in ts:
            q_interp = (1 - s) * q_start + s * q_end
            if not collision_checker(q_interp):
                collision_free = False
                break

        if collision_free:
            # Remove intermediate points
            path = np.vstack([path[:i+1], path[j:]])

    #build min length path
    MIN_WAYPOINTS = 5
    if len(path) < MIN_WAYPOINTS:
        ts = np.linspace(0, 1, MIN_WAYPOINTS)
        path = np.vstack([ (1-s)*path[0] + s*path[-1] for s in ts ])

    return path

def downsample_path(path, stride=3):
    return path[::stride]