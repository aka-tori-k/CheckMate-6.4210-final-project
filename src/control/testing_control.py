import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from control.execute_pp_traj import execute_trajectory
from simulation_setup.initialize_simulation import initialize_simulation
from path_planning.full_path_planning import compute_path
from path_planning.rrt_connect import  sample_random_q

def analyze_tracking(
    t_state, x_state,
    t_des, x_des,
    t_tau, torques,
    ):
    """
    x_state : (14, N) actual [q; v]
    x_des   : (14, N) desired [q; v]
    """

    q_actual = x_state[:7, :]
    qdot_actual = x_state[7:, :]

    q_des = x_des[:7, :]
    qdot_des = x_des[7:, :]

    # --- Interpolate desired signals onto actual timestamps ---
    q_des_interp = np.array([np.interp(t_state, t_des, q_des[i]) for i in range(7)])
    q_error = q_actual - q_des_interp

    # --- Joint error RMS + max ---
    rms_error = np.sqrt(np.mean(q_error**2, axis=1))
    max_error = np.max(np.abs(q_error), axis=1)

    print("\n=== Joint Tracking Error ===")
    for i in range(7):
        print(f"Joint {i+1}:  RMS = {rms_error[i]:.4f} rad,   Max = {max_error[i]:.4f} rad")

    # --- Plot joint error ---
    plt.figure()
    for i in range(7):
        plt.plot(t_state, q_error[i], label=f"Joint {i+1}")
    plt.title("Joint Position Tracking Error")
    plt.xlabel("Time (s)")
    plt.ylabel("q_actual - q_des (rad)")
    plt.legend()
    plt.grid(True)

    # --- Torque plot ---
    plt.figure()
    for i in range(7):
        plt.plot(t_tau, torques[i], label=f"τ{i+1}")
    plt.title("Control Torques")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)

    plt.show()

    return rms_error, max_error


def compute_ee_error(t_state, q_actual, q_des_interp, plant, context):
    iiwa = plant.GetModelInstanceByName("iiwa7")
    frame_EE = plant.GetFrameByName("iiwa_link_7")

    ee_err = []
    for k in range(q_actual.shape[1]):
        # actual
        plant.SetPositions(context, iiwa, q_actual[:,k])
        X_act = frame_EE.CalcPoseInWorld(context).translation()

        # desired 
        plant.SetPositions(context, iiwa, q_des_interp[:,k])
        X_des = frame_EE.CalcPoseInWorld(context).translation()

        ee_err.append(np.linalg.norm(X_act - X_des))

    ee_err = np.array(ee_err)

    print(f"\nEnd-effector RMS error: {np.sqrt(np.mean(ee_err**2)):.4f} m")
    print(f"End-effector MAX error: {np.max(ee_err):.4f} m")

    plt.figure()
    plt.plot(t_state, ee_err)
    plt.title("End-Effector Tracking Error")
    plt.xlabel("Time (s)")
    plt.ylabel("||x_act - x_des|| (m)")
    plt.grid(True)
    plt.show()

    return ee_err

if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat2, diagram, traj_source, logger_state, logger_desired, logger_torque = initialize_simulation(traj=None)
    
    q_goal = sample_random_q(plant)
    traj_pp = compute_path(
        q_goal=q_goal,
        plant=plant,
        scene_graph=scene_graph,
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
    
    (
    t_state, x_state,
    t_des, x_des,
    t_tau, torques
    ) = execute_trajectory(traj_pp, simulator, traj_source, diagram_context, logger_state, logger_desired, logger_torque)

    rms, maxerr = analyze_tracking(
        t_state, x_state,
        t_des, x_des,
        t_tau, torques,
        )

    ee_err = compute_ee_error(
        t_state,
        x_state[:7,:],
        x_des[:7,:],
        plant,
        plant.CreateDefaultContext()
    )


