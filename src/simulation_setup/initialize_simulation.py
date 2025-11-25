from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, MeshcatVisualizer,
    StartMeshcat, Simulator, Parser, MakeRenderEngineVtk, RenderEngineVtkParams, 
    MeshcatPointCloudVisualizer, DepthImageToPointCloud, 
    BaseField, CameraInfo, PixelType, RigidTransform, RotationMatrix
)
from add_pieces_to_plant import add_pieces_to_plant
from set_initial_piece_poses import set_initial_piece_poses
from add_robot_and_gripper import add_robot_and_gripper
from add_rgbd_sensor import add_rgbd_sensor
from grasp_planning.estimate_piece_pose import estimate_piece_pose
import numpy as np
import json
import os

def load_grasp_library():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "grasp_planning", "grasp_library.json")

    with open(path, "r") as f:
        data = json.load(f)

    # Convert dicts → RigidTransforms
    grasp_library = {}
    for piece_type, grasp_list in data.items():
        transforms = []
        for g in grasp_list:
            R = RotationMatrix(np.array(g["rotation"]))
            p = np.array(g["translation"])
            transforms.append(RigidTransform(R, p))
        grasp_library[piece_type] = transforms

    return grasp_library

def transform_grasp(T_world_piece, grasp):
    """
    grasp is a dict:
    {
        "contact_left": [...],
        "contact_right": [...],
        "approach": [...],
        "score": float
    }
    """
    cl = np.array(grasp["contact_left"] + [1])
    cr = np.array(grasp["contact_right"] + [1])
    ap = np.array(grasp["approach"] + [1])

    cl_w = T_world_piece @ cl
    cr_w = T_world_piece @ cr
    ap_w = T_world_piece @ ap
    
    return {
        "contact_left_w": cl_w[:3],
        "contact_right_w": cr_w[:3],
        "approach_w": ap_w[:3],
        "score": grasp["score"],
    }


def initialize_simulation():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    scene_graph.AddRenderer(
    "vtk",
    MakeRenderEngineVtk(RenderEngineVtkParams())
    )

    board = parser.AddModels("src/models/checkerboard_8_8_0_1/checkerboard.sdf")[0]
    plant, instances = add_pieces_to_plant(plant, parser)
    plant, iiwa, wsg = add_robot_and_gripper(plant, parser)
    plant, rgbd = add_rgbd_sensor(builder, plant, parser, scene_graph, iiwa)

    pc_gen = builder.AddSystem(
        DepthImageToPointCloud(
            camera_info=CameraInfo(width=640, height=480, fov_y=np.pi/4),
            pixel_type=PixelType.kDepth32F, 
            scale=1.0,
            fields = BaseField.kXYZs | BaseField.kRGBs
            )
        )
        
    # Connect images → pc generator
    builder.Connect(rgbd.GetOutputPort("depth_image_32f"), pc_gen.depth_image_input_port())
    builder.Connect(rgbd.GetOutputPort("color_image"), pc_gen.color_image_input_port())
    builder.Connect(rgbd.GetOutputPort("body_pose_in_world"), pc_gen.GetInputPort("camera_pose"))

    # Meshcat point cloud visualizer
    pc_vis = builder.AddSystem(
    MeshcatPointCloudVisualizer(meshcat, "rgbd_cloud", publish_period=0.1)
    )
    builder.Connect(pc_gen.point_cloud_output_port(), pc_vis.cloud_input_port())
        
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    plant.Finalize()

    diagram = builder.Build()
    simulator = Simulator(diagram)
    diagram_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    plant = set_initial_piece_poses(plant, plant_context, instances)

    robot_initial_pose = np.array([
        1.5,    # joint 1
        -1,     # joint 2
        0.0,    # joint 3
        -1.5,   # joint 4
        0.0,    # joint 5
        1.8,    # joint 6
        1.5     # joint 7
        ])

    plant.SetPositions(plant_context, iiwa, robot_initial_pose)

    grasp_library = load_grasp_library()

    simulator.Initialize()

    # Advance very slightly so camera outputs point cloud
    simulator.AdvanceTo(0.01)

    diagram_context = simulator.get_context()

    pc_context = pc_gen.GetMyContextFromRoot(diagram_context)

    # Get point cloud from the generator
    pc_msg = pc_gen.point_cloud_output_port().Eval(pc_context)
    pc_np = pc_msg.xyzs().T  # (N,3)

    # Test call
    piece_type = "pawn"
    square = "b7"
    T_world_board = np.eye(4)

    T_world_piece = estimate_piece_pose(pc_np, piece_type, square, T_world_board)
    print("Estimated pose:\n", T_world_piece)
    xyz = T_world_piece[:3, 3]
    show_sphere(meshcat, "debug/icp_sphere", xyz)

    # Load grasps for this piece type
    grasps = grasp_library[piece_type]

    # Transform all grasps to world frame
    grasps_world = [transform_grasp(T_world_piece, g) for g in grasps]

    # Rank by score (descending)
    grasps_world_sorted = sorted(grasps_world, key=lambda x: -x["score"])

    # Visualize top K grasps (let's use 8)
    TOP_K = 8
    for i, g_w in enumerate(grasps_world_sorted[:TOP_K]):
        show_grasp(meshcat, f"debug/grasp_{i}", g_w)

    print(f"Visualized {TOP_K} grasps for {piece_type}")
    print(meshcat.web_url())

    return simulator, plant, plant_context, meshcat


from pydrake.geometry import Sphere
from pydrake.geometry import Rgba

def show_sphere(meshcat, name, xyz, radius=0.01):
    meshcat.SetObject(name, Sphere(radius), rgba=Rgba(1, 0, 0, 1))
    meshcat.SetTransform(name, RigidTransform(xyz))

def show_grasp(meshcat, name_prefix, grasp_w):
    show_sphere(meshcat, f"{name_prefix}/cl", grasp_w["contact_left_w"], radius=0.015)
    show_sphere(meshcat, f"{name_prefix}/cr", grasp_w["contact_right_w"], radius=0.015)
    show_sphere(meshcat, f"{name_prefix}/ap", grasp_w["approach_w"], radius=0.01)



if __name__ == "__main__":
    simulator, plant, plant_context, meshcat = initialize_simulation()
    while True:
        pass
