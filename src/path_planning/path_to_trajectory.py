from pydrake.all import InverseDynamicsController, AddMultibodyPlantSceneGraph, TrajectorySource
from pydrake.trajectories import PiecewisePolynomial
import numpy as np
import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.initialize_simulation import initialize_simulation
from path_planning.rrt_connect import IiwaProblem, rrt_connect_planning, sample_random_q

def path_to_trajectory(path, dt=0.05):
    times = [i*dt for i in range(len(path))]
    knots = np.array(path).T  # shape (7, N)
    traj = PiecewisePolynomial.FirstOrderHold(times, knots)
    return traj

def add_velocities_to_path(path, dt):
    q = np.array(path)                    # (N, 7)
    qdot = np.gradient(q, dt, axis=0)     # numerical derivative (N, 7)
    full_traj = np.hstack([q, qdot])      # (N, 14)
    return full_traj

def path_to_fullstate_trajectory(path, dt=0.05):
    q_qdot = add_velocities_to_path(path, dt)     # (N, 14)
    times = [i*dt for i in range(len(path))]
    knots = q_qdot.T                               # (14, N)
    traj = PiecewisePolynomial.FirstOrderHold(times, knots)
    return traj


if __name__ == "__main__":
    #init sim
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram = initialize_simulation()
    #get a path
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram = initialize_simulation()
    q_start = sample_random_q(plant)
    q_goal  = sample_random_q(plant)

    print("Random start:", q_start)
    print("Random goal:", q_goal)
    
    problem = IiwaProblem(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.02,
        is_visualizing=True,
        plant=plant,
        scene_graph=scene_graph,
        diagram_context=diagram_context,
        plant_context=plant_context,
    )
    path, iters = rrt_connect_planning(problem, max_iterations=5000, eps_connect=0.05)

    traj = path_to_trajectory(path, dt=0.05)
    print(traj)

