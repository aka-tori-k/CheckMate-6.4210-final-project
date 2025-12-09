import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.initialize_simulation import initialize_simulation
from path_planning.rrt_connect import  sample_random_q
from path_planning.full_path_planning import compute_path
import time

def execute_trajectory(traj, simulator, traj_source, diagram_context, logger_state, logger_desired, logger_torque):
    # Upload the trajectory into the running modifiable TrajectorySource
    print("Uploading trajectory to controller...")
    traj_source.set_trajectory(traj)
    print(traj.start_time(), traj.end_time())
    # Simulate forward to the end of the trajectory (relative to current sim time)
    ctx = simulator.get_context()
    t_now = ctx.get_time()
    t_end = traj.end_time() + 0.01  # small buffer after end
    print(f"Advancing simulation from t={t_now:.3f} to t={t_end:.3f} ...")
    simulator.AdvanceTo(t_end)
    # print(simulator.get_context().get_time())

    # small pause (optional) before next plan
    # time.sleep(0.2)

    ###############################################################################
    # Everythign below is just for logging data out of the simulation for testing #
    ###############################################################################

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
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source, logger_state, logger_desired, logger_torque, pc_gen, wsg = initialize_simulation(traj=None)
    q_goal = sample_random_q(plant)
    traj_pp, q_spline = compute_path(
        q_goal=q_goal,
        plant=plant,
        scene_graph=scene_graph,
        diagram=diagram,
        diagram_context=diagram_context,
        plant_context=plant_context,
        meshcat=meshcat,
        draw_path=True,
        downsample=False,
        max_iterations=5000,
        eps_connect=0.05,
        num_iterations=200,
        time_per_waypoint=0.1,
        sample_dt=0.01
    )

    
    times_state, x_state, times_des, x_des, times_tau, torques = execute_trajectory(traj_pp, simulator, traj_source, 
                                                                                    diagram_context, logger_state, 
                                                                                    logger_desired, logger_torque
                                                                                    )