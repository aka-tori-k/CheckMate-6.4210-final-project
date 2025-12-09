from pydrake.all import RigidTransform

SQUARE = 0.1      # your board square size (meters)
BOARD_HALF = 4 * SQUARE   # 8 squares / 2 = 4

def square_to_pose(file_letter, rank_number, height=0.01):
    """
    Returns a RigidTransform placing a piece on a chess square.
    file_letter: 'a' to 'h'
    rank_number: '1' to '8'
    height: piece z-position (visual board is at z=0)
    """
    file_index = ord(file_letter.lower()) - ord('a')   # a=0, b=1, ...
    rank_index = int(rank_number) - 1                      # 1→0, 8→7

    x = (file_index + 0.5) * SQUARE - BOARD_HALF
    y = (rank_index + 0.5) * SQUARE - BOARD_HALF
    z = height

    return RigidTransform([x, y, z])

