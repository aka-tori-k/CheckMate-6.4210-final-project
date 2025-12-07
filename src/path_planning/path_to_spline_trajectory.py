from pydrake.all import PiecewisePolynomial
import numpy as np

def path_to_fullstate_trajectory(
    path,
    time_per_waypoint=0.02,
    sample_dt=0.01
):
    """
    Convert path (N x nq) into a smooth joint trajectory using
    PiecewisePolynomial.CubicHermite interpolation.
    
    Returns:
        traj_pp   : PiecewisePolynomial.FirstOrderHold for controller
        q_spline  : PiecewisePolynomial.CubicHermite for querying q, qdot, qdd analytically
    """
    path = np.asarray(path)
    N, nq = path.shape

    # Total duration
    T = time_per_waypoint * (N - 1)

    # Waypoint times
    ts = np.linspace(0, T, N)

    # Joint positions
    q_samples = path.T  # shape (nq, N)

    # Velocity at waypoints = 0 (Hermite)
    # Finite differences to approximate joint velocities between waypoints
    qdot_samples = np.zeros_like(q_samples)
    qdot_samples[:, :-1] = (q_samples[:, 1:] - q_samples[:, :-1]) / time_per_waypoint
    qdot_samples[:, -1] = qdot_samples[:, -2]  # copy last velocity


    # CubicHermite spline
    q_spline = PiecewisePolynomial.CubicHermite(ts, q_samples, qdot_samples)

    # Sample fine grid for FirstOrderHold
    num_samples = int(np.ceil(T / sample_dt)) + 1
    ts_fine = np.linspace(0, T, num_samples)

    qs = np.zeros((nq, num_samples))
    qdots = np.zeros((nq, num_samples))

    for k, t in enumerate(ts_fine):
        qs[:, k] = q_spline.value(t).reshape(nq)
        qdots[:, k] = q_spline.EvalDerivative(t).reshape(nq)

    # Full state [q; qdot]
    full = np.vstack([qs, qdots])  # shape (2*nq, num_samples)

    # Controller-friendly FirstOrderHold trajectory
    traj_pp = PiecewisePolynomial.FirstOrderHold(ts_fine, full)

    return traj_pp, q_spline