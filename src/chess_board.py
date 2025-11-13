
import time
import numpy as np

# Drake imports
from pydrake.geometry import (
    Box,
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
)
from pydrake.multibody.tree import (
    SpatialInertia,
    UnitInertia,
)
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph,
    MeshcatVisualizer, StartMeshcat,
    Simulator, RigidTransform
)
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder


# ---------------------
# Global Parameters
# ---------------------
SQUARE_SIZE = 0.1  # meters per square (6 cm)
BOARD_THICKNESS = 0.02
BOARD_SIZE = SQUARE_SIZE * 8
BOARD_Z = 0.0  # z of board top surface (we'll place board center at z = -BOARD_THICKNESS/2)

# Colors (RGBA)
WHITE_TILE = [1.0, 0.9, 0.8, 1.0]
BLACK_TILE = [0.2, 0.3, 0.3, 1.0]


def build_chess_board():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    # create a simple fixed board body
    # Give it small nonzero mass and unit inertia so we can register visual geometry easily,
    # then weld it to the world to keep it fixed.
    board_mass = 1.0
    board_half = BOARD_SIZE / 2.0
    # UnitInertia for a box: UnitInertia.SolidBox(x, y, z) may be available. If not, use diagonal inertia.
    try:
        unit_inertia = UnitInertia.SolidBox(BOARD_SIZE, BOARD_SIZE, BOARD_THICKNESS)
    except Exception:
        unit_inertia = UnitInertia(1.0, 1.0, 1.0)
    board_inertia = SpatialInertia(board_mass, np.zeros(3), unit_inertia)
    board_body = plant.AddRigidBody("chess_board", board_inertia)
    
    # Weld board to world frame
    plant.WeldFrames(plant.world_frame(), board_body.body_frame(), RigidTransform())

    # Register a large thin box as the board visual geometry
    board_visual_box = Box(BOARD_SIZE, BOARD_SIZE, BOARD_THICKNESS)
    
    # place the box so that top is at z = 0
    board_pose = RigidTransform([0, 0, -BOARD_THICKNESS / 2.0])
    plant.RegisterVisualGeometry(
        board_body,
        board_pose,
        board_visual_box,
        "board_visual",
        [0.85, 0.75, 0.6, 1.0],
    )

    # Create tiles as flat boxes (one visual per square)
    tile_thickness = 0.005
    tile_z_offset = BOARD_THICKNESS / 2.0 + tile_thickness / 2.0  # sit on top of board
    tile_box = Box(SQUARE_SIZE, SQUARE_SIZE, tile_thickness)
    files = "abcdefgh"
    ranks = "12345678"
    for fi, file in enumerate(files):
        for ri, rank in enumerate(ranks):
            x = (fi + 0.5) * SQUARE_SIZE - BOARD_SIZE / 2.0
            y = (ri + 0.5) * SQUARE_SIZE - BOARD_SIZE / 2.0
            tile_pose = RigidTransform([x, y, tile_z_offset])
            color = WHITE_TILE if ((fi + ri) % 2 == 0) else BLACK_TILE
            plant.RegisterVisualGeometry(
                board_body,
                tile_pose,
                tile_box,
                f"tile_{file}{rank}",
                color,
            )

    plant.Finalize()

    meshcat = StartMeshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()
    simulator.Initialize()
    simulator.AdvanceTo(0.01)

    return {
        "diagram": diagram,
        "simulator": simulator,
        "context": context,
        "meshcat": meshcat,
        "plant": plant,
        "scene_graph": scene_graph,
    }


if __name__ == "__main__":
    env = build_chess_board()
    print("Meshcat:", env["meshcat"].web_url())
    try:
        # Keep process alive so MeshCat/Drake remain accessible.
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting.")
