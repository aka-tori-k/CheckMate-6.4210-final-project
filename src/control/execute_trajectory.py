import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")

from path_planning.rrt_connect import rrt_connect_planning, IiwaProblem, sample_random_q

from pydrake.trajectories import PiecewisePolynomial
import numpy as np
import time

from simulation_setup.initialize_simulation import initialize_simulation

def downsample_path(path, stride=3):
    return path[::stride]

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


def execute_trajectory(traj, simulator, traj_source, diagram_context, logger_state, logger_desired, logger_torque):
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

    log_state_context = logger_state.GetMyContextFromRoot(diagram_context)
    log_state = logger_state.GetLog(log_state_context)
    times_state = log_state.sample_times()
    x_state = log_state.data()  # shape (14, N)

    # Desired State Log
    log_desired_context = logger_desired.GetMyContextFromRoot(diagram_context)
    log_desired = logger_desired.GetLog(log_desired_context)
    times_des = log_desired.sample_times()
    x_des = log_desired.data()  # shape (14, N)

    # Torque Log
    log_torque_context = logger_torque.GetMyContextFromRoot(diagram_context)
    log_torque = logger_torque.GetLog(log_torque_context)
    times_tau = log_torque.sample_times()
    torques = log_torque.data()  # shape (7, N)
    return times_state, x_state, times_des, x_des, times_tau, torques
    
if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat2, diagram, traj_source, logger_state, logger_desired, logger_torque = initialize_simulation(traj=None)
    # You should sample a start state from the *current* plant positions if you want realistic planning
    # Get current robot positions:
    iiwa = plant.GetModelInstanceByName("iiwa7")
    q_start = plant.GetPositions(plant_context, iiwa)
    q_goal = sample_random_q(plant)

    # collision free start and goal for testing
    # q_start = np.array([1.14481155, -1.1725643, 0.74546698, -0.5089159, -2.85271485, 0.85927073, 0.44859717]) 
    # q_goal = np.array([0.29921316, -0.8316761, 0.59264661, -1.32333652, 0.16495925, -0.93726397, -1.58362034])
    
    # print("Current robot positions (q_start):", q_start)

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

    #debugging prints
    print(problem.start_in_collision)
    print(problem.goal_in_collision)
    
    path, iters = rrt_connect_planning(problem, max_iterations=5000, eps_connect=0.05)
    path = downsample_path(path, stride=3) #higher stride is faster path
    problem.draw_path(path, plant, diagram_context, plant_context,meshcat)
    traj = path_to_fullstate_trajectory(path, dt=0.005) #lower dt is faster execution
    times_state, x_state, times_des, x_des, times_tau, torques = execute_trajectory(traj, simulator, traj_source, diagram_context, logger_state, logger_desired, logger_torque)
    print("State log shape:", x_state.shape)
    print("Desired log shape:", x_des.shape)
    print("Torque log shape:", torques.shape)