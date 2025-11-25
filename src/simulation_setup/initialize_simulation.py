from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, MeshcatVisualizer,
    StartMeshcat, Simulator, Parser, MakeRenderEngineVtk, RenderEngineVtkParams, 
    MeshcatPointCloudVisualizer, DepthImageToPointCloud, 
    BaseField, CameraInfo, PixelType, RigidTransform, RotationMatrix
)
import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.add_pieces_to_plant import add_pieces_to_plant
from simulation_setup.set_initial_piece_poses import set_initial_piece_poses
from simulation_setup.add_robot_and_gripper import add_robot_and_gripper
from simulation_setup.add_rgbd_sensor import add_rgbd_sensor
import numpy as np

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

    #initial pose for path planning tests
    # robot_initial_pose = [0, 0, 0, -1.57, 0, 1.57, 0]

    plant.SetPositions(plant_context, iiwa, robot_initial_pose)

    simulator.Initialize()

    print(meshcat.web_url())

    return simulator, plant, plant_context, meshcat

if __name__ == "__main__":
    simulator, plant, plant_context, meshcat = initialize_simulation()
    while True:
        pass
