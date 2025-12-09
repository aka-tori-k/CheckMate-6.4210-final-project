from simulation_setup.initialize_simulation import initialize_simulation
from chess_interface import ChessInterface
from simulation_setup.square_to_pose import square_to_pose
from pydrake.all import Box, Rgba, RigidTransform
from grasp_planning.calculate_grasp import calculate_grasp
from grasp_planning.load_grasp_library import load_grasp_library
from path_planning.full_path_planning import compute_path
from path_planning.generate_iiwa_problem import generate_iiwa_problem
from path_planning.path_to_spline_trajectory import shift_trajectory
from path_planning.get_next_square_pose import get_next_square_pose
from control.execute_pp_traj import execute_trajectory
import numpy as np


def reset_robot_and_sync(plant, plant_context, diagram_context, diagram, q_target):
    """
    Reset robot to target position and sync geometry for collision checks.
    
    This is important because path planning and collision checking may modify
    the robot's position, so we need to reset it before executing trajectories.
    """
    iiwa = plant.GetModelInstanceByName("iiwa7")
    plant.SetPositions(plant_context, iiwa, q_target)
    diagram.ForcedPublish(diagram_context)


def draw_square_box(meshcat, square, color, box_name, height=0.005, box_size=0.095):
    """
    Draw a colored box around a chess square.
    
    Args:
        meshcat: Meshcat visualizer instance
        square: Chess square notation (e.g., "e2")
        color: Rgba color tuple
        box_name: Unique name for the box in meshcat
        height: Height of the box above the board (default 5mm)
        box_size: Size of the box (default 95mm, slightly smaller than 100mm square)
    """
    # Parse square notation (e.g., "e2" -> file='e', rank='2')
    file_letter = square[0].lower()
    rank_number = square[1]
    
    # Get the center position of the square
    square_pose = square_to_pose(file_letter, rank_number, height=height)
    square_center = square_pose.translation()
    
    # Create a box slightly smaller than the square
    box = Box(box_size, box_size, 0.001)  # Thin box (2mm thick)
    
    # Set the box object and transform in meshcat
    meshcat.SetObject(box_name, box, rgba=color)
    meshcat.SetTransform(box_name, RigidTransform(square_center))


def find_valid_grasp_poses(plant, plant_context, diagram_context, diagram, scene_graph, meshcat, pc_gen, wsg, 
                          grasp_library, piece_type, square, max_valid_poses=10):
    """
    Calculate grasp poses for a piece and filter for collision-free poses.
    
    Returns:
        valid_poses: List of valid (collision-free) joint angle configurations
    """
    print(f"Calculating grasp poses for {piece_type} on {square}...")
    iiwa = plant.GetModelInstanceByName("iiwa7")
    q_start = plant.GetPositions(plant_context, iiwa)
    
    joint_angles = calculate_grasp(plant, plant_context, diagram_context, meshcat, pc_gen, wsg, 
                                   grasp_library, piece_type, square, show_pose=True)
    
    if joint_angles is None or len(joint_angles) == 0:
        print(f"No grasp poses found for {piece_type} on {square}")
        return []
    
    # Reset robot to starting position after calculate_grasp (may have modified position)
    reset_robot_and_sync(plant, plant_context, diagram_context, diagram, q_start)
    
    print(f"Found {len(joint_angles)} possible grasps, checking for collision-free poses...")
    
    valid_poses = []
    for i, pose in enumerate(joint_angles):
        problem, is_collision_free = generate_iiwa_problem(
            q_start=q_start,
            q_goal=pose,
            plant=plant,
            scene_graph=scene_graph,
            diagram=diagram,
            diagram_context=diagram_context,
            plant_context=plant_context
        )
        
        if not problem.start_in_collision and not problem.goal_in_collision:
            print(f"Found valid pose {i+1}/{len(joint_angles)}")
            valid_poses.append(pose)
            if len(valid_poses) >= max_valid_poses:
                break
    
    print(f"Found {len(valid_poses)} valid collision-free grasp poses")
    return valid_poses


def plan_trajectory(q_goal, plant, plant_context, diagram_context, diagram, 
                   meshcat, scene_graph,
                   max_iterations=5000, eps_connect=0.05, num_iterations=200,
                   time_per_waypoint=0.5, sample_dt=0.0001, draw_path=True):
    """
    Plan a path to the goal pose.
    
    Returns:
        traj_pp: PiecewisePolynomial trajectory, or None if planning failed
        q_spline: Spline for querying, or None if planning failed
    """
    try:
        print(f"Planning path to goal pose...")
        traj_pp, q_spline = compute_path(
            q_goal=q_goal,
            plant=plant,
            scene_graph=scene_graph,
            diagram=diagram,
            diagram_context=diagram_context,
            plant_context=plant_context,
            meshcat=meshcat,
            draw_path=draw_path,
            downsample=False,
            max_iterations=max_iterations,
            eps_connect=eps_connect,
            num_iterations=num_iterations,
            time_per_waypoint=time_per_waypoint,
            sample_dt=sample_dt
        )
        print("Path planning successful")
        return traj_pp, q_spline
    except Exception as e:
        print(f"Path planning failed: {e}")
        return None, None


def execute_planned_trajectory(traj_pp, plant, plant_context, diagram_context, diagram,
                               simulator, traj_source,
                               logger_state, logger_desired, logger_torque,
                               q_start=None):
    """
    Execute a planned trajectory.
    
    Args:
        traj_pp: PiecewisePolynomial trajectory to execute
        q_start: Starting position to reset to before execution (if None, uses current position)
    
    Returns:
        success: bool indicating if execution succeeded
        current_q: Current joint angles after execution
    """
    if traj_pp is None:
        print("Cannot execute: trajectory is None")
        return False, None
    
    iiwa = plant.GetModelInstanceByName("iiwa7")
    
    # Reset robot to starting position and sync geometry (path planning may have modified position)
    if q_start is None:
        q_start = plant.GetPositions(plant_context, iiwa)
    reset_robot_and_sync(plant, plant_context, diagram_context, diagram, q_start)
    
    # Shift trajectory to current simulation time
    traj_pp = shift_trajectory(traj_pp, simulator.get_context().get_time())
    
    # Execute trajectory
    print("Executing trajectory...")
    execute_trajectory(traj_pp, simulator, traj_source, diagram_context, 
                      logger_state, logger_desired, logger_torque)
    
    current_q = plant.GetPositions(plant_context, iiwa)
    print(f"Trajectory execution complete. Current joint angles: {current_q}")
    
    return True, current_q


def main():
    # 1) initialize simulation
    (simulator, plant, plant_context, meshcat, scene_graph, 
    diagram_context, meshcat, diagram, traj_source,
    logger_state, logger_desired, logger_torque, pc_gen, wsg) = initialize_simulation()
   
   # 2) initialize chess interface and get best move
    chess_interface = ChessInterface()
    move_info = chess_interface.get_best_move()
    move = move_info['move']
    from_square = move_info['from_square']
    to_square = move_info['to_square']
    moving_type = move_info['moving_type']
    piece_name = move_info['piece_name']
    is_capture = move_info['is_capture']
    captured_type = move_info['captured_type']
    print(f"Best move: {move}, from {from_square} to {to_square}, moving {moving_type} ({piece_name}), capture: {is_capture}, captured type: {captured_type}")
    
    # Visualize the move: green box on source square, red box on target square
    draw_square_box(meshcat, from_square, Rgba(0, 1, 0, 0.7), "source_square_box")
    draw_square_box(meshcat, to_square, Rgba(1, 0, 0, 0.7), "target_square_box")
    print(f"Visualized: Green box on {from_square}, Red box on {to_square}")

    # Advance simulation to let things settle
    simulator.AdvanceTo(1.0)
    
    # Load grasp library
    grasp_library = load_grasp_library()
    
    # Get scene_graph from diagram
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    
    # Save initial robot pose
    iiwa = plant.GetModelInstanceByName("iiwa7")
    initial_pose = plant.GetPositions(plant_context, iiwa)
    
    # ===== Execute move workflow =====
    
    # Step 1: Find valid grasp poses
    print("\n=== Step 1: Finding valid grasp poses ===")
    valid_poses = find_valid_grasp_poses(
        plant, plant_context, diagram_context, diagram, scene_graph, meshcat, pc_gen, wsg,
        grasp_library, moving_type, from_square, max_valid_poses=10
    )
    
    if len(valid_poses) == 0:
        print("No valid grasp poses found. Cannot proceed.")
        chess_interface.close_engine()
        return
    
    # Step 2: Plan and execute to first valid grasp pose
    print("\n=== Step 2: Moving to grasp pose ===")
    # Reset robot to current position before planning (collision checks may have modified it)
    reset_robot_and_sync(plant, plant_context, diagram_context, diagram, initial_pose)
    
    traj_pp, _ = plan_trajectory(
        valid_poses[0], plant, plant_context, diagram_context, diagram,
        meshcat, scene_graph
    )
    
    if traj_pp is None:
        print("Failed to plan path to grasp pose")
        chess_interface.close_engine()
        return
    
    success, current_q = execute_planned_trajectory(
        traj_pp, plant, plant_context, diagram_context, diagram,
        simulator, traj_source, logger_state, logger_desired, logger_torque,
        q_start=initial_pose
    )
    
    if not success:
        print("Failed to execute trajectory to grasp pose")
        chess_interface.close_engine()
        return
    
    # Step 3: TODO - Grasp piece (gripper actuation)
    print("\n=== Step 3: TODO - Grasp piece ===")
    
    # Step 4: Get target square pose
    print(f"\n=== Step 4: Computing target square pose for {to_square} ===")
    target_pose = get_next_square_pose(to_square, piece_name, plant, plant_context, wsg)
    
    if target_pose is None:
        print(f"Failed to compute target pose for {to_square}")
        chess_interface.close_engine()
        return
    
    # Step 5: Plan and execute to target square
    print(f"\n=== Step 5: Moving to target square {to_square} ===")
    # Reset robot to current position before planning
    reset_robot_and_sync(plant, plant_context, diagram_context, diagram, current_q)
    
    traj_pp, _ = plan_trajectory(
        target_pose, plant, plant_context, diagram_context, diagram,
        meshcat, scene_graph
    )
    
    if traj_pp is None:
        print("Failed to plan path to target square")
        chess_interface.close_engine()
        return
    
    success, current_q = execute_planned_trajectory(
        traj_pp, plant, plant_context, diagram_context, diagram,
        simulator, traj_source, logger_state, logger_desired, logger_torque,
        q_start=current_q  # Use current_q from previous execution
    )
    
    if not success:
        print("Failed to execute trajectory to target square")
        chess_interface.close_engine()
        return
    
    # Step 6: TODO - Release piece (gripper actuation)
    print("\n=== Step 6: TODO - Release piece ===")
    
    # Step 7: Return to initial pose
    print("\n=== Step 7: Returning to initial pose ===")
    # Reset robot to current position before planning
    reset_robot_and_sync(plant, plant_context, diagram_context, diagram, current_q)
    
    traj_pp, _ = plan_trajectory(
        initial_pose, plant, plant_context, diagram_context, diagram,
        meshcat, scene_graph
    )
    
    if traj_pp is None:
        print("Failed to plan path to initial pose")
        chess_interface.close_engine()
        return
    
    success, _ = execute_planned_trajectory(
        traj_pp, plant, plant_context, diagram_context, diagram,
        simulator, traj_source, logger_state, logger_desired, logger_torque,
        q_start=current_q  # Use current_q from previous execution
    )
    
    if not success:
        print("Failed to execute trajectory to initial pose")
        chess_interface.close_engine()
        return
    
    # Update internal board state
    print("\n=== Move complete - updating board state ===")
    chess_interface.apply_move(move)
    print("Board state updated successfully")
    
    chess_interface.close_engine()

if __name__ == "__main__":
    main()