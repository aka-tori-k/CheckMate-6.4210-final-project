import sys

from pyparsing import col
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from pydrake.all import (Sphere, Rgba, RigidTransform)
from simulation_setup.initialize_simulation import initialize_simulation
from grasp_planning.new_estimate_piece_pose import estimate_piece_pose
from grasp_planning.solve_ik_for_grasp import solve_ik_for_grasp
from grasp_planning.load_grasp_library import load_grasp_library
from pydrake.all import Meshcat
import numpy as np

def show_frame(meshcat, path, T, axis_length=0.1, line_width=3):
    R = T.rotation().matrix()
    p = T.translation()

    # Create endpoints for each axis
    X = p.reshape(3,1)
    Y = p.reshape(3,1) + R[:,0:1] * axis_length
    Z = p.reshape(3,1) + R[:,1:2] * axis_length
    W = p.reshape(3,1) + R[:,2:3] * axis_length

    # Meshcat requires Fortran contiguous!
    def F(x):
        return np.asfortranarray(x)

    meshcat.SetLineSegments(f"{path}/x", F(X), F(Y), line_width=line_width, rgba=Rgba(1,0,0,1))
    meshcat.SetLineSegments(f"{path}/y", F(X), F(Z), line_width=line_width, rgba=Rgba(0,1,0,1))
    meshcat.SetLineSegments(f"{path}/z", F(X), F(W), line_width=line_width, rgba=Rgba(0,0,1,1))



def show_sphere(meshcat, name, xyz, radius=0.01):
    meshcat.SetObject(name, Sphere(radius), rgba=Rgba(1, 0, 0, 1))
    meshcat.SetTransform(name, RigidTransform(xyz))

def calculate_grasp(plant, plant_context, diagram_context, meshcat, pc_gen, wsg, grasp_library, piece_type, square, show_pose=False):
    pc_context = pc_gen.GetMyContextFromRoot(diagram_context)

    # Get point cloud from the generator
    pc_msg = pc_gen.point_cloud_output_port().Eval(pc_context)
    pc_np = pc_msg.xyzs().T  # Nx3
    T_world_piece = estimate_piece_pose(pc_np, piece_type, square)
    if show_pose:
        xyz = T_world_piece.GetAsMatrix4()[:3, 3]
        show_sphere(meshcat, "debug/icp_sphere", xyz)
        print(f"Estimated pose for {piece_type} on {square}:\n{T_world_piece.GetAsMatrix4()}")
    
    # return []
    grasps = grasp_library[piece_type]
    joint_angles = []

    print(f"Number of grasps to try: {len(grasps)}")
    #change to enumerate
    for i, T_object_grasp in enumerate(grasps):
        print(f"Trying IK on grasp {i+1}/{len(grasps)} \r")
        grasp_in_world = T_world_piece @ T_object_grasp 
        q = solve_ik_for_grasp(plant, plant_context, wsg, grasp_in_world)
        if q is not None:
            joint_angles.append(q)
        
    if not joint_angles:
        print("No poses found")
        return
    return joint_angles


    
if __name__ == "__main__":
    # simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source, logger_state, logger_desired, logger_torque, pc_gen, wsg = initialize_simulation()
    # simulator.AdvanceTo(1)  # let things settle
    # iiwa = plant.GetModelInstanceByName("iiwa7")
    # grasp_library = load_grasp_library()
    # piece_type = "pawn"
    # square = "f2"
    
    # joint_angles = calculate_grasp(plant, plant_context, diagram_context, meshcat, pc_gen, wsg, grasp_library, piece_type, square, show_pose=True)
    # print(f"Found {len(joint_angles)} possible grasps for {piece_type} on {square}")

    # joint1_angles = joint_angles[0]
    # print(f"First grasp joint angles: \n {joint1_angles}")
    # plant.SetPositions(plant_context, iiwa, joint1_angles)
    # # print('here')
    # # print(f'current iiwa positions: \n {plant.GetPositions(plant_context, iiwa)}')
    # diagram.ForcedPublish(diagram_context)
    while True:
        pass

