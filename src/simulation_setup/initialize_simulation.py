from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, MeshcatVisualizer,
    StartMeshcat, Simulator, RigidTransform, Parser
)
from add_pieces_to_plant import add_pieces_to_plant
from set_initial_piece_poses import set_initial_piece_poses
from add_robot_and_gripper import add_robot_and_gripper

def initialize_simulation():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    board = parser.AddModels("src/models/checkerboard_8_8_0_1/checkerboard.sdf")[0]
    plant, instances = add_pieces_to_plant(plant, parser)
    plant, iiwa, wsg = add_robot_and_gripper(plant, parser)

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    plant.Finalize()

    diagram = builder.Build()
    simulator = Simulator(diagram)
    diagram_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    plant = set_initial_piece_poses(plant, plant_context, instances)

    simulator.Initialize()

    print(meshcat.web_url())

    return simulator, plant, plant_context, meshcat

if __name__ == "__main__":
    simulator, plant, plant_context, meshcat = initialize_simulation()
    while True:
        pass
