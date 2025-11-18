
def add_pieces_to_plant(plant, parser):
    """
    Adds chess pieces to the plant from their SDF files.
    """
    piece_files = {
        'black_bishop1': "src/models/pieces/bishops/black_bishop1.sdf",
        'black_bishop2': "src/models/pieces/bishops/black_bishop2.sdf",
        'black_knight1': "src/models/pieces/knights/black_knight1.sdf",
        'black_knight2': "src/models/pieces/knights/black_knight2.sdf",
        'black_rook1': "src/models/pieces/rooks/black_rook1.sdf",
        'black_rook2': "src/models/pieces/rooks/black_rook2.sdf",
        'black_queen': "src/models/pieces/queens/black_queen.sdf",
        'black_king': "src/models/pieces/kings/black_king.sdf",
        'black_pawn1': "src/models/pieces/pawns/black_pawn1.sdf",
        'black_pawn2': "src/models/pieces/pawns/black_pawn2.sdf",
        'black_pawn3': "src/models/pieces/pawns/black_pawn3.sdf",
        'black_pawn4': "src/models/pieces/pawns/black_pawn4.sdf",
        'black_pawn5': "src/models/pieces/pawns/black_pawn5.sdf",
        'black_pawn6': "src/models/pieces/pawns/black_pawn6.sdf",
        'black_pawn7': "src/models/pieces/pawns/black_pawn7.sdf",   
        'black_pawn8': "src/models/pieces/pawns/black_pawn8.sdf",

        'white_bishop1': "src/models/pieces/bishops/white_bishop1.sdf",
        'white_bishop2': "src/models/pieces/bishops/white_bishop2.sdf",
        'white_knight1': "src/models/pieces/knights/white_knight1.sdf",
        'white_knight2': "src/models/pieces/knights/white_knight2.sdf",
        'white_rook1': "src/models/pieces/rooks/white_rook1.sdf",
        'white_rook2': "src/models/pieces/rooks/white_rook2.sdf",
        'white_queen': "src/models/pieces/queens/white_queen.sdf",
        'white_king': "src/models/pieces/kings/white_king.sdf",
        'white_pawn1': "src/models/pieces/pawns/white_pawn1.sdf",
        'white_pawn2': "src/models/pieces/pawns/white_pawn2.sdf",
        'white_pawn3': "src/models/pieces/pawns/white_pawn3.sdf",
        'white_pawn4': "src/models/pieces/pawns/white_pawn4.sdf",
        'white_pawn5': "src/models/pieces/pawns/white_pawn5.sdf",
        'white_pawn6': "src/models/pieces/pawns/white_pawn6.sdf",
        'white_pawn7': "src/models/pieces/pawns/white_pawn7.sdf",
        'white_pawn8': "src/models/pieces/pawns/white_pawn8.sdf",
    }
    instances = {}
    for piece_name, file_path in piece_files.items():
        model_instance = parser.AddModels(file_path)[0]
        instances[piece_name] = model_instance
    return plant, instances