from pydrake.all import PiecewisePolynomial
import numpy as np


def shift_trajectory(traj_pp, shift, hold_end_time=0.5):
    """Shift trajectory forward by `shift` seconds, preserving final hold."""
    old_times = traj_pp.get_segment_times()
    new_times = [t + shift for t in old_times]

    # Values at each knot
    values = np.hstack([traj_pp.value(t) for t in old_times])

    # Add explicit hold
    if hold_end_time > 0:
        new_times.append(new_times[-1] + hold_end_time)
        values = np.hstack([values, values[:, -1][:, None]])

    return PiecewisePolynomial.FirstOrderHold(new_times, values)



def path_to_fullstate_trajectory(
    path,
    time_per_waypoint=0.02,
    sample_dt=0.002,
    hold_end_time=0.5
):
    """
    Convert an Nx7 joint path into a smooth, controller-safe full-state trajectory.
    Produces:
        - Cubic Hermite spline for querying smooth q(t), v(t)
        - FOH full-state trajectory [q; v] for IDC tracking
    """
    path = np.asarray(path)
    N, nq = path.shape

    # -------------------------------
    # 1) Waypoint times
    # -------------------------------
    T = time_per_waypoint * (N - 1)
    ts = np.linspace(0, T, N)

    q_samples = path.T                                 # (nq, N)
    qdot_samples = np.zeros_like(q_samples)            # (nq, N)

    # -------------------------------
    # 2) Velocities via finite diff
    # -------------------------------
    if N > 2:
        qdot_samples[:, 1:-1] = (
            q_samples[:, 2:] - q_samples[:, :-2]
        ) / (2 * time_per_waypoint)

    qdot_samples[:, 0] = 0.0     # Start at rest
    qdot_samples[:, -1] = 0.0    # End at rest (IMPORTANT)

    # -------------------------------
    # 3) Smooth cubic Hermite spline
    # -------------------------------
    q_spline = PiecewisePolynomial.CubicHermite(ts, q_samples, qdot_samples)

    # -------------------------------
    # 4) FOH sampling grid
    # -------------------------------
    T_end = T + hold_end_time
    num_samples = int(np.ceil(T_end / sample_dt)) + 1
    ts_fine = np.linspace(0, T_end, num_samples)

    qs = np.zeros((nq, num_samples))
    qdots = np.zeros((nq, num_samples))

    # -------------------------------
    # 5) Sample spline, then HOLD
    # -------------------------------
    for k, t in enumerate(ts_fine):
        if t <= T:
            qs[:, k] = q_spline.value(t).flatten()
            qdots[:, k] = q_spline.EvalDerivative(t).flatten()
        else:
            # Exact hold at final pose
            qs[:, k] = q_samples[:, -1]
            qdots[:, k] = 0.0

    # -------------------------------
    # 6) Build full-state [q; v]
    # -------------------------------
    full = np.vstack([qs, qdots])   # shape (2*nq, num_samples)

    # -------------------------------
    # 7) FOH trajectory: controller-safe
    # -------------------------------
    traj_pp = PiecewisePolynomial.FirstOrderHold(ts_fine, full)

    return traj_pp, q_spline
