import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.initialize_simulation import initialize_simulation
from path_planning.rrt_connect import  sample_random_q
from path_planning.full_path_planning import compute_path
from control.execute_pp_traj import execute_trajectory
from grasp_planning.calculate_grasp import calculate_grasp
from grasp_planning.load_grasp_library import load_grasp_library
from path_planning.generate_iiwa_problem import generate_iiwa_problem
from path_planning.path_to_spline_trajectory import shift_trajectory
from path_planning.get_next_square_pose import get_next_square_pose

import numpy as np
import time

if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source, logger_state, logger_desired, logger_torque, pc_gen, wsg = initialize_simulation(traj=None)
    print("Initialization complete. Advancing simulation...")
    simulator.AdvanceTo(1)
    print("Simulation advanced successfully.")
    iiwa = plant.GetModelInstanceByName("iiwa7")
    real_q_start = plant.GetPositions(plant_context, iiwa)
    print(f"Current iiwa positions: {real_q_start}")

    grasp_library = load_grasp_library()
    piece_type = "pawn"
    square = "e2"

    joint_angles = calculate_grasp(plant, plant_context, diagram_context, meshcat, pc_gen, wsg, grasp_library, piece_type, square, show_pose= True)
    print(f"Found {len(joint_angles)} possible grasps for {piece_type} on {square}")
    
    plant.SetPositions(plant_context, iiwa, real_q_start)

    # IMPORTANT: sync all geometry pose updates for collision checks
    diagram.ForcedPublish(diagram_context)

    q_start = plant.GetPositions(plant_context, iiwa)
    
    poses_in_collision = 0
    valid_poses = []
    for iteration, pose in enumerate(joint_angles):
        problem, is_collision_free = generate_iiwa_problem(
            q_start=q_start,
            q_goal=pose,
            plant=plant,
            scene_graph=scene_graph,
            diagram=diagram,
            diagram_context=diagram_context,
            plant_context=plant_context
        )
        # print(f"Start in collision: {problem.start_in_collision}")
        # print(f"Goal in collision: {problem.goal_in_collision}")

        if not problem.start_in_collision and not problem.goal_in_collision:
            print(f"Found valid pose at iteration {iteration}")
            valid_poses.append(pose)
            # print(f"Valid pose: {pose}")
            if iteration >= 10:  # limit to first 5 valid poses
                break
        else:
            poses_in_collision += 1

    # reset iiwa to starting position
    plant.SetPositions(plant_context, iiwa, real_q_start)

    # IMPORTANT: sync all geometry pose updates for collision checks
    diagram.ForcedPublish(diagram_context)


    print(f"Number of valid poses found: {len(valid_poses)}")
    for i, pose in enumerate(valid_poses):
        try:
            traj_pp, q_spline = compute_path(
                q_goal=pose,
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
                time_per_waypoint=0.5,
                sample_dt=0.0001
            )
            print(f"Path planning successful for valid pose {pose} (at index {i})")
            break  # exit loop if path planning is successful
        except Exception as e:
            print(f"Path planning failed for this pose with error: {e}")
            continue

    # reset iiwa to starting position
    plant.SetPositions(plant_context, iiwa, real_q_start)

    # IMPORTANT: sync all geometry pose updates for collision checks
    diagram.ForcedPublish(diagram_context)
    traj_pp = shift_trajectory(traj_pp, simulator.get_context().get_time())

    times_state, x_state, times_des, x_des, times_tau, torques = execute_trajectory(
        traj_pp, simulator, traj_source,
        diagram_context, logger_state,
        logger_desired, logger_torque
    )

    current_q = plant.GetPositions(plant_context, iiwa)
    print(f"Current iiwa positions after executing trajectory: {current_q}")
    #print type of current_q
    # print(f"Type of current_q: {type(current_q)}")
    
    time.sleep(5)

    temp = []

    next_joint_angles = get_next_square_pose("e4", "white_pawn5", plant, plant_context, wsg)
    print(f"Next joint angles to reach e4 for white_pawn5:\n{next_joint_angles}")
    # print(f"Type of next_joint_angles: {type(next_joint_angles)}")
    temp.append(next_joint_angles)
    for i, pose in enumerate(temp):
        try:
            traj_pp, q_spline = compute_path(
                q_goal=pose,
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
                time_per_waypoint=0.5,
                sample_dt=0.0001
            )
            print(f"Path planning successful for valid pose {pose} (at index {i})")
            break  # exit loop if path planning is successful
        except Exception as e:
            print(f"Path planning failed for this pose with error: {e}")
            continue

    # reset iiwa to starting position
    plant.SetPositions(plant_context, iiwa, current_q)

    # IMPORTANT: sync all geometry pose updates for collision checks
    diagram.ForcedPublish(diagram_context)

    traj_pp = shift_trajectory(traj_pp, simulator.get_context().get_time())

    times_state, x_state, times_des, x_des, times_tau, torques = execute_trajectory(
        traj_pp, simulator, traj_source,
        diagram_context, logger_state,
        logger_desired, logger_torque
    )


    current_q = plant.GetPositions(plant_context, iiwa)
    print(f"Current iiwa positions after executing second trajectory: {current_q}")
    #print type of current_q
    # print(f"Type of current_q: {type(current_q)}")
    
    time.sleep(5)
    temp = []
    temp.append(real_q_start)
    for i, pose in enumerate(temp):
        try:
            traj_pp, q_spline = compute_path(
                q_goal=pose,
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
                time_per_waypoint=0.5,
                sample_dt=0.0001
            )
            print(f"Path planning successful for valid pose {pose} (at index {i})")
            break  # exit loop if path planning is successful
        except Exception as e:
            print(f"Path planning failed for this pose with error: {e}")
            continue
    
    # reset iiwa to starting position
    plant.SetPositions(plant_context, iiwa, current_q)

    # IMPORTANT: sync all geometry pose updates for collision checks
    diagram.ForcedPublish(diagram_context)

    traj_pp = shift_trajectory(traj_pp, simulator.get_context().get_time())

    times_state, x_state, times_des, x_des, times_tau, torques = execute_trajectory(
        traj_pp, simulator, traj_source,
        diagram_context, logger_state,
        logger_desired, logger_torque
    )


    while True:
        pass