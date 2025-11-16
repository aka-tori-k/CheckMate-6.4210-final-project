import time
from manipulation.scenarios import AddIiwa, AddWsg
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, MeshcatVisualizer,
    StartMeshcat, Simulator, RigidTransform, Parser
)

meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)

# --- Load board ---
board = parser.AddModels("src/models/checkerboard_8_8_0_1/checkerboard.sdf")[0]
board_frame = plant.GetFrameByName("checkerboard_8_8_0_1_body", board)

# --- Load IIWA manually ---
iiwa = parser.AddModelsFromUrl(
    "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
)[0]

iiwa_base = plant.GetFrameByName("iiwa_link_0", iiwa)

iiwa_pose = RigidTransform([0, -0.75, 0])   # behind board
plant.WeldFrames(plant.world_frame(), iiwa_base, iiwa_pose)

# --- Load WSG gripper ---
wsg = AddWsg(plant, iiwa, roll=0.0, welded=True)

# finalizing + visualization
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
plant.Finalize()

diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()

print(meshcat.web_url())
while True:
    time.sleep(1)
