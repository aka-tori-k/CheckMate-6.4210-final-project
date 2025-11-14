from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph,
                         MeshcatVisualizer, StartMeshcat,
                         Simulator, Parser)
from load_simulation import load_chessboard_from_yaml
import time

meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)

# Load chessboard with pieces
load_chessboard_from_yaml(plant, parser, "src/piece_positions.yaml")

plant.Finalize()

# Add visualizer
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# Build and run simulation
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()


print("Chessboard visualization running in MeshCat. Press Ctrl+C to exit.")
print(f"Open: {meshcat.web_url()}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")