import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.square_to_pose import square_to_pose
from grasp_planning.solve_ik_for_grasp import solve_ik_for_grasp
import time

def get_next_square_pose(goal_square, piece_name, plant, plant_context, wsg):
    """
    Arguments:
    - goal_square: str, e.g. "e4"
    - piece_name: str, e.g. "white_pawn5"
    - plant: MultibodyPlant
    - plant_context: Context for the plant
    - wsg: gripper model instance

    Returns:
    - joint_angles_goal: np.array of joint angles to reach the goal square pose
    """
    #NEED TO GET piece_name FROM INTERNAL REPRESENTATION
    file_letter = goal_square[0].lower()
    rank_number = goal_square[1]
    piece_instance = plant.GetModelInstanceByName(piece_name)
    X_W_piece = square_to_pose(file_letter, rank_number)
    X_piece_gripper = plant.CalcRelativeTransform(plant_context, 
                                                  plant.GetFrameByName(piece_name, piece_instance),
                                                  plant.GetFrameByName("body_frame", wsg),)
    X_W_gripper = X_W_piece @ X_piece_gripper
    joint_angles_goal = solve_ik_for_grasp(plant, plant_context, wsg, X_W_gripper)
    return joint_angles_goal

if __name__ == "__main__":
    from simulation_setup.initialize_simulation import initialize_simulation
    from grasp_planning.calculate_grasp import calculate_grasp
    from grasp_planning.load_grasp_library import load_grasp_library
     

    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source, logger_state, logger_desired, logger_torque, pc_gen, wsg = initialize_simulation()

    grasp_library = load_grasp_library()
    piece_type = "pawn"
    square = "e2"

    joint_angles = calculate_grasp(plant, plant_context, diagram_context, meshcat, pc_gen, wsg, grasp_library, piece_type, square, show_pose= True)

    print(joint_angles)
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa7"), joint_angles[0])
    diagram.ForcedPublish(diagram_context)
    time.sleep(2)

    goal_square = "e4"
    piece = "white_pawn5" #will need to query internal rep later for this
    print(wsg)
    joint_angles_goal = get_next_square_pose(goal_square, piece, plant, plant_context, wsg)
    print(f"Joint angles to reach {goal_square} for {piece}:\n{joint_angles_goal}")
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa7"), joint_angles_goal)
    diagram.ForcedPublish(diagram_context)
    while True:
        pass




