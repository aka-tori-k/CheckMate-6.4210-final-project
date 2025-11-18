from square_to_pose import square_to_pose

def set_initial_piece_poses(plant, context, instances):
    """
    Sets pose of pieces on board based on predefined mapping
    """
    piece_files = {
        'black_bishop1': ('c', 8),
        'black_bishop2': ('f', 8),
        'black_knight1': ('b', 8),
        'black_knight2': ('g', 8),
        'black_rook1': ('a', 8),
        'black_rook2': ('h', 8),
        'black_queen': ('d', 8),
        'black_king': ('e', 8),
        'black_pawn1': ('a', 7),
        'black_pawn2': ('b', 7),
        'black_pawn3': ('c', 7),
        'black_pawn4': ('d', 7),
        'black_pawn5': ('e', 7),
        'black_pawn6': ('f', 7),
        'black_pawn7': ('g', 7),   
        'black_pawn8': ('h', 7),

        'white_bishop1': ('c', 1),
        'white_bishop2': ('f', 1),
        'white_knight1': ('b', 1),
        'white_knight2': ('g', 1),
        'white_rook1': ('a', 1),
        'white_rook2': ('h', 1),
        'white_queen': ('d', 1),
        'white_king': ('e', 1),
        'white_pawn1': ('a', 2),
        'white_pawn2': ('b', 2),
        'white_pawn3': ('c', 2),
        'white_pawn4': ('d', 2),
        'white_pawn5': ('e', 2),
        'white_pawn6': ('f', 2),
        'white_pawn7': ('g', 2),
        'white_pawn8': ('h', 2),    
    }
    
    for piece_name, (file_letter, rank_number) in piece_files.items():
        model_instance = instances[piece_name]
        body = plant.GetBodyByName(piece_name, model_instance)

        piece_pose = square_to_pose(file_letter, rank_number)
        plant.SetFreeBodyPose(context, body, piece_pose)
    return plant