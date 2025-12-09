import sys
import time
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from path_planning.rrt_connect import rrt_connect_planning 
from path_planning.shortcutting import downsample_path, shortcut_path
from path_planning.path_to_spline_trajectory import path_to_fullstate_trajectory
from path_planning.generate_iiwa_problem import generate_iiwa_problem
import numpy as np
from simulation_setup.initialize_simulation import initialize_simulation

def compute_path(q_goal, plant, scene_graph, diagram, diagram_context, plant_context, meshcat, draw_path=False, downsample=False,
                 max_iterations=5000, eps_connect=0.05,num_iterations=200 ,time_per_waypoint=0.2, sample_dt=0.01):
    
    #get robot and its current pose
    iiwa = plant.GetModelInstanceByName("iiwa7")
    q_start = plant.GetPositions(plant_context, iiwa)

    # collision free start and goal for testing
    # q_start = np.array([1.14481155, -1.1725643, 0.74546698, -0.5089159, 0.85927073, 0.44859717]) 
    # q_goal = np.array([0.29921316, -0.8316761, 0.59264661, -1.32333652, 0.16495925, -0.93726397, -1.58362034])

    # debugging prints
    # print("Current robot positions (q_start):", q_start)
    # print("q_goal:", q_goal)

    #geenerate planning problem
    problem, is_collision_free = generate_iiwa_problem(
        q_start=q_start,
        q_goal=q_goal,
        plant=plant,
        scene_graph=scene_graph,
        diagram=diagram,
        diagram_context=diagram_context,
        plant_context=plant_context
    )

    #debugging prints
    # print(f"Start in collision: {problem.start_in_collision}")
    # print(f"Goal in collision: {problem.goal_in_collision}")

    # print("Current robot positions (q_start):", q_start)
    # print("problem.goal:", problem.goal)


    #plan rrt-connect path
    path, iters = rrt_connect_planning(problem, max_iterations=max_iterations, eps_connect=eps_connect)
    #prnit first and last ofo path for debugging
    # print("Planned path start:", path[0])
    # print("Planned path end:", path[-1])

    #shortcut path
    path_shortcut = shortcut_path(path, collision_checker=is_collision_free, num_iterations=num_iterations)

    #ensure robot at right point at start of path
    # plant.SetPositions(plant_context, iiwa, path_shortcut[0])
    # # plant.SetVelocities(plant_context, iiwa, np.zeros(7))

    # # Optional: downsample decrease waypoints (prolly not needed)
    # if downsample:
    #     path_shortcut = downsample_path(path_shortcut, stride=2)

    # # Optional: visualize path in meshcat
    if draw_path:
        problem.draw_path(path_shortcut, plant, diagram_context, plant_context,meshcat)

    # # 4. Generate trajectory
    traj_pp, q_spline = path_to_fullstate_trajectory(path_shortcut, time_per_waypoint=time_per_waypoint, sample_dt=sample_dt)

    return traj_pp, q_spline


if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source, logger_state, logger_desired, logger_torque, pc_gen, wsg = initialize_simulation(traj=None)
    q_goal = np.array([0.29921316, -0.8316761, 0.59264661, -1.32333652, 0.16495925, -0.93726397, -1.58362034])
    path_shortcut = compute_path(
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

    print(f"Computed path with {len(path_shortcut)} waypoints.")