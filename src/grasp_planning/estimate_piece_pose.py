import numpy as np
from pydrake.all import RigidTransform
from pydrake.math import RotationMatrix

from manipulation.icp import IterativeClosestPoint
import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.square_to_pose import square_to_pose
import trimesh

def load_mesh_as_points(path, num_samples=5000):
    mesh = trimesh.load(path)
    pts, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return pts

def estimate_piece_pose(pc, piece_type, square, T_world_board):
    """
    Estimate the 6D pose of a chess piece from a point cloud.

    Args:
        pc: (N,3) numpy array of point cloud in world frame
        piece_type: string: 'pawn', 'rook', 'knight', etc
        square: string: 'e8'
        T_world_board: 4x4 transform of board frame in world frame

    Returns:
        T_world_piece: 4x4 homogeneous transform of the piece
    """

    file_letter, rank_number = square

    T_world_board = RigidTransform(T_world_board)

    # Get the world coordinates of square center
    T_board_square = square_to_pose(file_letter, rank_number)
    square_center_world = (T_world_board @ T_board_square).translation()

    SQUARE_SIZE = 0.1
    half = SQUARE_SIZE / 2

    xmin = square_center_world[0] - half
    xmax = square_center_world[0] + half
    ymin = square_center_world[1] - half
    ymax = square_center_world[1] + half

    # Crop world-frame point cloud to the square
    X = pc[:, 0]
    Y = pc[:, 1]

    mask = (
        (X >= xmin) & (X <= xmax) &
        (Y >= ymin) & (Y <= ymax)
    )
    cropped = pc[mask]

    if cropped.shape[0] < 30:
        raise RuntimeError("Not enough points on that square to fit ICP.")

    # Transform cropped cloud into BOARD frame 
    T_board_world = T_world_board.inverse()   # world → board
    pc_board = np.array([
        T_board_world @ np.hstack([point, 1.0])
        for point in cropped
    ])[:, :3]

    # Load the mesh corresponding to the piece type
    obj_path = {
        "pawn":   "src/models/pieces/pawns/pawn_mesh.obj",
        "rook":   "src/models/pieces/rooks/rook_mesh.obj",
        "bishop": "src/models/pieces/bishops/bishop_mesh.obj",
        "knight": "src/models/pieces/knights/knight_mesh.obj",
        "queen":  "src/models/pieces/queens/queen_mesh.obj",
        "king":   "src/models/pieces/kings/king_mesh.obj",
    }[piece_type]

    # load mesh
    mesh_pts = load_mesh_as_points(obj_path)
    mesh = trimesh.load(obj_path)
    print(mesh.bounds)
    print(mesh.scale)
    print(mesh.extents)

    # Run ICP in BOARD frame
    # mesh_pts: model points in board frame
    # pc_board: observed points in board frame
    X_WO_hat, icp_info = IterativeClosestPoint(mesh_pts.T, pc_board.T)

    return X_WO_hat, cropped


SQUARE = 0.1      # your board square size (meters)
BOARD_HALF = 4 * SQUARE   # 8 squares / 2 = 4

