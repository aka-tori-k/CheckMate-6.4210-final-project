from simulation_setup.initialize_simulation import initialize_simulation
from chess_interface import ChessInterface


def main():
    # 1) initialize simulation
    (simulator, plant, plant_context, meshcat, scene_graph, 
    diagram_context, meshcat, diagram, traj_source,
    logger_state, logger_desired, logger_torque) = initialize_simulation()
   
   # 2) initialize chess interface and get best move
    chess_interface = ChessInterface()
    move, from_square, to_square, moving_type, is_capture, captured_type = chess_interface.get_best_move().values()
    print(f"Best move: {move}, from {from_square} to {to_square}, moving {moving_type}, capture: {is_capture}, captured type: {captured_type}")

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