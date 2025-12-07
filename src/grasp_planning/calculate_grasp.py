import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from pydrake.all import (Sphere, Rgba, RigidTransform)
from simulation_setup.initialize_simulation import initialize_simulation
from grasp_planning.new_estimate_piece_pose import estimate_piece_pose
from grasp_planning.solve_ik_for_grasp import solve_ik_for_grasp
from grasp_planning.load_grasp_library import load_grasp_library

def show_sphere(meshcat, name, xyz, radius=0.01):
    meshcat.SetObject(name, Sphere(radius), rgba=Rgba(1, 0, 0, 1))
    meshcat.SetTransform(name, RigidTransform(xyz))

def calculate_grasp(diagram_context, pc_gen, wsg, grasp_library, piece_type, square, show_pose=False):
    pc_context = pc_gen.GetMyContextFromRoot(diagram_context)

    # Get point cloud from the generator
    pc_msg = pc_gen.point_cloud_output_port().Eval(pc_context)
    pc_np = pc_msg.xyzs().T  # Nx3
    T_world_piece = estimate_piece_pose(pc_np, piece_type, square)
    if show_pose:
        xyz = T_world_piece.GetAsMatrix4()[:3, 3]
        show_sphere(meshcat, "debug/icp_sphere", xyz)
    grasps = grasp_library[piece_type]
    joint_angles = []
    for T_object_grasp in grasps:
        grasp_in_world = T_world_piece @ T_object_grasp
        q = solve_ik_for_grasp(plant, plant_context, wsg, grasp_in_world)
        if q is not None:
            joint_angles.append(q)
    if not joint_angles:
        print("No poses found")
        return
    return joint_angles

    
if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source, logger_state, logger_desired, logger_torque, pc_gen, wsg = initialize_simulation()
    grasp_library = load_grasp_library()
    piece_type = "pawn"
    square = "c2"
    print(calculate_grasp(diagram_context, pc_gen, wsg, grasp_library, piece_type, square, True))


