import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")

from path_planning.rrt_connect import rrt_connect_planning, IiwaProblem, sample_random_q

from pydrake.trajectories import PiecewisePolynomial
import numpy as np
import time

from simulation_setup.initialize_simulation import initialize_simulation


def path_to_fullstate_trajectory(path, dt=0.05):
    """
    Convert path (N x nq) into a PiecewisePolynomial whose value(t) = [q; qdot].
    """
    path = np.asarray(path)   # shape (N, nq)
    print("Path shape:", path.shape)
    print(path)
    N, nq = path.shape
    times = np.linspace(0.0, dt * (N - 1), N)
    qdot = np.gradient(path, dt, axis=0)
    full_knots = np.hstack([path, qdot]).T   # shape (2*nq, N)
    traj = PiecewisePolynomial.FirstOrderHold(times, full_knots)
    return traj


def execute_trajectory(traj, simulator, traj_source):
    # Upload the trajectory into the running modifiable TrajectorySource
    print("Uploading trajectory to controller...")
    traj_source.set_trajectory(traj)

    # Simulate forward to the end of the trajectory (relative to current sim time)
    ctx = simulator.get_context()
    t_now = ctx.get_time()
    t_end = t_now + traj.end_time()
    print(f"Advancing simulation from t={t_now:.3f} to t={t_end:.3f} ...")
    simulator.AdvanceTo(t_end)

    # small pause (optional) before next plan
    time.sleep(0.2)

    
if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat2, diagram, traj_source = initialize_simulation(traj=None)
    # You should sample a start state from the *current* plant positions if you want realistic planning
    # Get current robot positions:
    iiwa = plant.GetModelInstanceByName("iiwa7")
    # q_now = plant.GetPositions(plant_context, iiwa)

    # # sample a new random goal (or use any target you like)
    # q_goal = sample_random_q(plant)
    q_start = np.array([ 1.14481155, -1.1725643, 0.74546698, -0.5089159, -2.85271485, 0.85927073, 0.44859717]) 
    q_goal = np.array([ 0.29921316, -0.8316761, 0.59264661, -1.32333652, 0.16495925, -0.93726397, -1.58362034])

    problem = IiwaProblem(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.02,
        is_visualizing=False,
        plant=plant,
        scene_graph=scene_graph,
        diagram_context=diagram_context,
        plant_context=plant_context,
    )
    
    
    path, iters = rrt_connect_planning(problem, max_iterations=5000, eps_connect=0.05)
    problem.draw_path(path, plant, diagram_context, plant_context,meshcat)
    traj = path_to_fullstate_trajectory(path, dt=0.05)
    execute_trajectory(traj, simulator, traj_source)