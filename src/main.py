from simulation_setup.initialize_simulation import initialize_simulation
from chess_interface import ChessInterface
from simulation_setup.square_to_pose import square_to_pose
from pydrake.all import Box, Rgba, RigidTransform


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
    box = Box(box_size, box_size, 0.002)  # Thin box (2mm thick)
    
    # Set the box object and transform in meshcat
    meshcat.SetObject(box_name, box, rgba=color)
    meshcat.SetTransform(box_name, RigidTransform(square_center))


def main():
    # 1) initialize simulation
    (simulator, plant, plant_context, meshcat, scene_graph, 
    diagram_context, meshcat, diagram, traj_source,
    logger_state, logger_desired, logger_torque, pc_gen, wsg) = initialize_simulation()
   
   # 2) initialize chess interface and get best move
    chess_interface = ChessInterface()
    move, from_square, to_square, moving_type, is_capture, captured_type = chess_interface.get_best_move().values()
    print(f"Best move: {move}, from {from_square} to {to_square}, moving {moving_type}, capture: {is_capture}, captured type: {captured_type}")
    
    # Visualize the move: green box on source square, red box on target square
    draw_square_box(meshcat, from_square, Rgba(0, 1, 0, 0.7), "source_square_box")
    draw_square_box(meshcat, to_square, Rgba(1, 0, 0, 0.7), "target_square_box")
    print(f"Visualized: Green box on {from_square}, Red box on {to_square}")

    simulator.AdvanceTo(15.0)

    # 3) compute_grasp_pose()
    
    # 4) compute_path() from current pose to grasp pose

    # 5) execute_path() to grasp pose

    # 6) grasp_piece()

    # 7) compute_path to target square pose

    # 8) execute_path to target square pose

    # 9) release_piece()

    # 10) return to initial pose

    # 11) update internal board state
    chess_interface.apply_move(move)  # update board
    
    chess_interface.close_engine()

if __name__ == "__main__":
    main()