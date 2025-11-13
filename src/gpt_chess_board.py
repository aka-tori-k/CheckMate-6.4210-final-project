#!/usr/bin/env python3
"""
drake_chess.py

Simple Drake + MeshCat chessboard demo.

Requirements:
- Drake installed (pydrake)
- meshcat python package (pip install meshcat) often comes with Drake installs
- Launch this script and open the MeshCat URL printed to the terminal.

Controls:
- Use the exposed functions at the bottom (after building the system) to set/move/remove pieces.
  Example usage (in same process / interactive session):
    set_piece("w_rook_a1", "a1")
    set_piece("b_pawn_a7", "a7")
    move_piece("w_rook_a1", "a4")
    remove_piece("b_pawn_a7")

Author: ChatGPT (GPT-5 Thinking mini)
"""

import time
import numpy as np

# Drake imports
from pydrake.geometry import (
    Box,
    Cylinder,
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
# Parameters
# ---------------------
SQUARE_SIZE = 0.06  # meters per square (6 cm)
BOARD_THICKNESS = 0.02
BOARD_SIZE = SQUARE_SIZE * 8
PIECE_BASE_RADIUS = 0.02
PIECE_HEIGHT = 0.04
BOARD_Z = 0.0  # z of board top surface (we'll place board center at z = -BOARD_THICKNESS/2)

# Colors (RGBA)
WHITE_TILE = [1.0, 0.9, 0.8, 1.0]
BLACK_TILE = [0.2, 0.3, 0.3, 1.0]
WHITE_PIECE = [0.95, 0.95, 0.95, 1.0]
BLACK_PIECE = [0.05, 0.05, 0.05, 1.0]


# ---------------------
# Helper functions
# ---------------------
def algebraic_to_xy(square: str):
    """
    Convert algebraic notation like 'a1'..'h8' to coordinates (x,y) in meters.
    We define a1 at the lower-left when viewing +x to the right and +y forward.
    Board center is at origin (0,0).
    """
    if len(square) != 2:
        raise ValueError("Square must be like 'a1'")
    file = square[0].lower()
    rank = square[1]
    files = "abcdefgh"
    ranks = "12345678"
    file_idx = files.index(file)
    rank_idx = ranks.index(rank)
    # coordinates of square center relative to board center
    x = (file_idx + 0.5) * SQUARE_SIZE - BOARD_SIZE / 2.0
    y = (rank_idx + 0.5) * SQUARE_SIZE - BOARD_SIZE / 2.0
    return x, y


# ---------------------
# Build Drake diagram
# ---------------------
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

# We'll keep a registry of pieces we add: dict piece_id -> dict(body, model_instance, name)
pieces = dict()


def add_piece(piece_id: str, color="white", square="a1"):
    """
    Create a rigid body (cylinder) to represent a chess piece and place it at `square`.
    piece_id must be unique.
    """
    if piece_id in pieces:
        raise ValueError(f"piece_id {piece_id} already exists")
    # create a small rigid body
    mass = 0.1
    # approximate inertia for a cylinder about center (use UnitInertia.SolidCylinder if available)
    try:
        inertia = UnitInertia.SolidCylinder(PIECE_HEIGHT / 2.0, PIECE_BASE_RADIUS)
    except Exception:
        inertia = UnitInertia(1.0, 1.0, 1.0)
    inertia3 = SpatialInertia(mass, np.zeros(3), inertia)
    body = plant.AddRigidBody(piece_id, inertia3)

    # Register visual geometry: cylinder with base at board top
    cyl = Cylinder(PIECE_HEIGHT, PIECE_BASE_RADIUS)
    x, y = algebraic_to_xy(square)
    # place the piece so its base sits on the tile surface
    piece_z = tile_z_offset + tile_thickness / 2.0 + PIECE_HEIGHT / 2.0
    pose = RigidTransform([x, y, piece_z])
    rgba = WHITE_PIECE if color == "white" else BLACK_PIECE
    plant.RegisterVisualGeometry(body, pose, cyl, f"{piece_id}_visual", rgba)
    # Optionally register collision geometry (same shape)
    try:
        plant.RegisterCollisionGeometry(body, pose, cyl, f"{piece_id}_collision", rgba)
    except Exception:
        # older/newer drake versions slightly differ on collision registration signature
        pass

    pieces[piece_id] = {"body": body, "pose": pose, "color": color, "square": square}
    return piece_id


def move_piece(piece_id: str, square: str):
    """
    Move the visual geometry of the piece to a new square.
    Note: This routine updates the registered visual geometry by
    re-registering a new visual with the same name (works in many Drake versions),
    and updates our stored pose. For more advanced dynamics you'd instead set the state.
    """
    if piece_id not in pieces:
        raise KeyError(f"No piece named {piece_id}")
    body = pieces[piece_id]["body"]
    x, y = algebraic_to_xy(square)
    piece_z = tile_z_offset + tile_thickness / 2.0 + PIECE_HEIGHT / 2.0
    new_pose = RigidTransform([x, y, piece_z])
    # Try re-registering visual geometry. Some Drake versions will allow re-registration;
    # otherwise one would need to modify the plant before Finalize (not possible).
    try:
        # Remove prior visual is not provided; so we register a new visual with a unique name
        new_name = f"{piece_id}_visual_moved_{int(time.time()*1000)}"
        cyl = Cylinder(PIECE_HEIGHT, PIECE_BASE_RADIUS)
        rgba = WHITE_PIECE if pieces[piece_id]["color"] == "white" else BLACK_PIECE
        plant.RegisterVisualGeometry(body, new_pose, cyl, new_name, rgba)
        pieces[piece_id]["pose"] = new_pose
        pieces[piece_id]["square"] = square
    except Exception as e:
        # As fallback, just update stored pose (if visualization doesn't update immediately it may require rebuilding)
        pieces[piece_id]["pose"] = new_pose
        pieces[piece_id]["square"] = square
        print("Warning: Could not register new visual geometry. Visualization may not reflect moves until restart.")
        print("Exception:", e)


def remove_piece(piece_id: str):
    """
    Remove a piece from our registry (note: Drake doesn't offer safe runtime removal of bodies;
    this function marks it removed; restarting may be required to free resources).
    """
    if piece_id not in pieces:
        raise KeyError(f"No piece named {piece_id}")
    # We can't reliably remove bodies from MultibodyPlant at runtime in most Drake versions.
    # So we merely mark it as removed; optionally we can register an invisible visual at far-away pose.
    pieces.pop(piece_id)


def list_pieces():
    return {k: {"square": v["square"], "color": v["color"]} for k, v in pieces.items()}

# Example utility to set up initial standard chess position (very basic, only pawns and rooks/knights/bishops/queen/king)
def setup_standard_position():
    clear_board()
    # white back rank
    add_piece("w_rook_a1", "white", "a1")
    add_piece("w_knight_b1", "white", "b1")
    add_piece("w_bishop_c1", "white", "c1")
    add_piece("w_queen_d1", "white", "d1")
    add_piece("w_king_e1", "white", "e1")
    add_piece("w_bishop_f1", "white", "f1")
    add_piece("w_knight_g1", "white", "g1")
    add_piece("w_rook_h1", "white", "h1")
    for fi, file in enumerate("abcdefgh"):
        add_piece(f"w_pawn_{file}2", "white", f"{file}2")

    # black back rank
    add_piece("b_rook_a8", "black", "a8")
    add_piece("b_knight_b8", "black", "b8")
    add_piece("b_bishop_c8", "black", "c8")
    add_piece("b_queen_d8", "black", "d8")
    add_piece("b_king_e8", "black", "e8")
    add_piece("b_bishop_f8", "black", "f8")
    add_piece("b_knight_g8", "black", "g8")
    add_piece("b_rook_h8", "black", "h8")
    for fi, file in enumerate("abcdefgh"):
        add_piece(f"b_pawn_{file}7", "black", f"{file}7")


def clear_board():
    # Note: can't truly remove bodies at runtime; however for simplicity we'll launch a fresh diagram if needed.
    pieces.clear()
    # A more robust implementation would rebuild the diagram entirely.


setup_standard_position()
# Finalize plant
plant.Finalize()

# Start MeshCat (automatically opens or reuses an instance)
meshcat = StartMeshcat()
print("MeshCat is available at:", meshcat.web_url())

# Add MeshCat visualizer
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_context()
# Initialize simulator a single step to push initial geometry to MeshCat
simulator.Initialize()
simulator.AdvanceTo(0.01)



# If the user runs this file as a script, we set up a sample position and give instructions.
if __name__ == "__main__":
    print("Building sample chess position...")
    # setup_standard_position()
    print("Initial pieces placed. You can use the following functions from this Python process:")
    print("  add_piece(piece_id, color='white'/'black', square='e4')")
    print("  move_piece(piece_id, square='e4')")
    print("  remove_piece(piece_id)")
    print("  list_pieces()")
    print("Open the MeshCat URL printed above to view the board.")
    print("This script will keep running -- interrupt (Ctrl-C) to exit.")

    try:
        # Keep process alive so MeshCat/Drake remain accessible.
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting.")
