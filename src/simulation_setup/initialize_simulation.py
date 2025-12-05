from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    MeshcatVisualizer,
    StartMeshcat,
    Simulator,
    Parser,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    MeshcatPointCloudVisualizer,
    DepthImageToPointCloud,
    BaseField,
    CameraInfo,
    PixelType,
    InverseDynamicsController,
    MultibodyPlant,
    ConstantVectorSource,
)
import numpy as np
import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")

from simulation_setup.add_pieces_to_plant import add_pieces_to_plant
from simulation_setup.set_initial_piece_poses import set_initial_piece_poses
from simulation_setup.add_robot_and_gripper import add_robot_and_gripper
from simulation_setup.add_rgbd_sensor import add_rgbd_sensor
from simulation_setup.ModifiableTrajectorySource import ModifiableTrajectorySource

def initialize_simulation(traj=None, realtime_rate=1.0, kp_scale=400.0, kd_scale=40.0):
    """
    Builds the full simulation diagram once, including:
      - world plant, pieces, board, iiwa, wsg, sensors
      - a controller-only MultibodyPlant (iiwa-only, welded)
      - an InverseDynamicsController wired between a ModifiableTrajectorySource and the world plant
    Returns:
      simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source
    traj (optional): a PiecewisePolynomial full-state ([q; qdot]) to initialize the traj source.
    """

    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))

    # Add board and pieces (your existing functions)
    board = parser.AddModels("src/models/checkerboard_8_8_0_1/checkerboard.sdf")[0]
    plant, instances = add_pieces_to_plant(plant, parser)
    plant, iiwa_instance, wsg = add_robot_and_gripper(plant, parser)
    plant, rgbd = add_rgbd_sensor(builder, plant, parser, scene_graph, iiwa_instance)

    # Depth -> pointcloud chain
    pc_gen = builder.AddSystem(
        DepthImageToPointCloud(
            camera_info=CameraInfo(width=640, height=480, fov_y=np.pi / 4),
            pixel_type=PixelType.kDepth32F,
            scale=1.0,
            fields=BaseField.kXYZs | BaseField.kRGBs,
        )
    )
    builder.Connect(rgbd.GetOutputPort("depth_image_32f"), pc_gen.depth_image_input_port())
    builder.Connect(rgbd.GetOutputPort("color_image"), pc_gen.color_image_input_port())
    builder.Connect(rgbd.GetOutputPort("body_pose_in_world"), pc_gen.GetInputPort("camera_pose"))

    pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, "rgbd_cloud", publish_period=0.1))
    builder.Connect(pc_gen.point_cloud_output_port(), pc_vis.cloud_input_port())

    # Meshcat visualizer (keeps your existing visualizer)
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Finalize world plant
    plant.Finalize()

    # ---------- Build controller-only plant (iiwa-only) ----------
    controller_plant = MultibodyPlant(time_step=0.0)
    controller_parser = Parser(controller_plant)

    # load same SDF as world (floating base) but weld base into controller plant
    ctrl_model = controller_parser.AddModelsFromUrl(
        "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
    )[0]

    controller_plant.WeldFrames(
        controller_plant.world_frame(),
        controller_plant.GetFrameByName("iiwa_link_0", ctrl_model)
    )
    controller_plant.Finalize()

    # nq = controller_plant.num_positions()
    # # full_state_len = controller_plant.num_multibody_positions() + controller_plant.num_multibody_velocities()
    # full_state_len = controller_plant.num_positions() + controller_plant.num_velocities()
    # assert full_state_len == 2 * nq, "controller plant expected full-state length == 2*nq"
    nq = controller_plant.num_positions()
    nv = controller_plant.num_velocities()
    full_state_len = nq + nv
    assert full_state_len == 2 * nq, "controller plant expected q+v = 2*nq"


    # ---------- Create a modifiable trajectory source and connect the controller ----------
    traj_source = ModifiableTrajectorySource(vector_size=full_state_len)
    if traj is not None:
        # quick consistency check
        if traj.value(traj.start_time()).size != full_state_len:
            raise RuntimeError("Provided traj has wrong output dimension for controller.")
        traj_source.set_trajectory(traj)

    traj_src_system = builder.AddSystem(traj_source)  # Add to builder and keep handle

    # Build IDC on controller_plant
    kp = kp_scale * np.ones(nq)
    kd = kd_scale * np.ones(nq)
    ki = np.zeros(nq)

    id_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant, kp, ki, kd, has_reference_acceleration=False)
    )

    # Wire trajectory source -> controller desired_state
    builder.Connect(traj_src_system.get_output_port(0), id_controller.get_input_port_desired_state())

    # Wire world plant state -> controller estimated_state
    builder.Connect(plant.get_state_output_port(iiwa_instance), id_controller.get_input_port_estimated_state())

    # Wire controller torque -> world plant actuation input
    builder.Connect(id_controller.get_output_port_control(), plant.get_actuation_input_port(iiwa_instance))

    # Optionally zero external generalized forces for safety
    try:
        zero_tau = ConstantVectorSource(np.zeros(nq))
        zsys = builder.AddSystem(zero_tau)
        builder.Connect(zsys.get_output_port(), plant.get_generalized_external_forces_input_port(iiwa_instance))
    except Exception:
        # Not fatal if the world plant doesn't expose that port
        pass

    # Build diagram + simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    diagram_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # set initial piece poses and robot initial pose (your existing logic)
    plant = set_initial_piece_poses(plant, plant_context, instances)

    # robot_initial_pose = np.array([
    #     1.5,    # joint 1
    #     -1,     # joint 2
    #     0.0,    # joint 3
    #     -1.5,   # joint 4
    #     0.0,    # joint 5
    #     1.8,    # joint 6
    #     1.5     # joint 7
    #     ])

    # initial pose for path planning tests
    # robot_initial_pose = [0, 0, 0, -1.57, 0, 1.57, 0]
    robot_initial_pose = [0, -0.5, 0, -1.0, 0, 1.0, 0]
    plant.SetPositions(plant_context, iiwa_instance, robot_initial_pose)

    simulator.Initialize()
    simulator.set_target_realtime_rate(realtime_rate)

    # Advance very slightly so camera outputs point cloud
    simulator.AdvanceTo(0.01)

    diagram_context = simulator.get_context()

    pc_context = pc_gen.GetMyContextFromRoot(diagram_context)

    # Get point cloud from the generator
    pc_msg = pc_gen.point_cloud_output_port().Eval(pc_context)
    pc_np = pc_msg.xyzs().T  # (N,3)

    # # Test call
    # piece_type = "pawn"
    # square = "b7"
    # T_world_board = np.eye(4)

    # T_world_piece = estimate_piece_pose(pc_np, piece_type, square, T_world_board)
    # print("Estimated pose:\n", T_world_piece)
    # xyz = T_world_piece[:3, 3]


    print(meshcat.web_url())

    # Return everything + the handle to the traj_source system so caller can set trajectories at runtime
    return simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source



if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram, traj_source = initialize_simulation()
    while True:
        pass