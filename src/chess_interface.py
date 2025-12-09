import chess
import chess.engine

PIECE_TYPE_MAP = {
    chess.PAWN: "pawn",
    chess.ROOK: "rook",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.QUEEN: "queen",
    chess.KING: "king"
}

class ChessInterface:
    """
    Interface to interact with the chess engine and board.
    
    Instance Variables:
    - engine_path: Path to the chess engine executable.
    - engine: The chess engine instance.
    - board: The current chess board state.
    - square_to_piece_name: Dictionary mapping square names (e.g., "e2") to piece names (e.g., "white_pawn5")
    
    Methods:
    - get_best_move(time_limit): Returns the best move and metadata.
    - apply_move(move): Applies a move to the internal board state.
    - is_game_over(): Checks if the game is over.
    - close_engine(): Closes the chess engine.
    """

    def __init__(self, engine_path="/usr/games/stockfish"):
        # ------- Engine + board setup -------
        self.engine_path = engine_path  # in the dev container
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.board = chess.Board()
        
        # Initialize mapping from squares to piece names (starting position)
        self.square_to_piece_name = self._initialize_piece_mapping()
    
    def _initialize_piece_mapping(self):
        """
        Initialize the mapping from squares to piece names based on starting position.
        Returns a dictionary mapping square names (e.g., "e2") to piece names (e.g., "white_pawn5").
        """
        # Starting positions: square -> piece_name
        return {
            'a8': 'black_rook1',
            'b8': 'black_knight1',
            'c8': 'black_bishop1',
            'd8': 'black_queen',
            'e8': 'black_king',
            'f8': 'black_bishop2',
            'g8': 'black_knight2',
            'h8': 'black_rook2',
            'a7': 'black_pawn1',
            'b7': 'black_pawn2',
            'c7': 'black_pawn3',
            'd7': 'black_pawn4',
            'e7': 'black_pawn5',
            'f7': 'black_pawn6',
            'g7': 'black_pawn7',
            'h7': 'black_pawn8',
            'a1': 'white_rook1',
            'b1': 'white_knight1',
            'c1': 'white_bishop1',
            'd1': 'white_queen',
            'e1': 'white_king',
            'f1': 'white_bishop2',
            'g1': 'white_knight2',
            'h1': 'white_rook2',
            'a2': 'white_pawn1',
            'b2': 'white_pawn2',
            'c2': 'white_pawn3',
            'd2': 'white_pawn4',
            'e2': 'white_pawn5',
            'f2': 'white_pawn6',
            'g2': 'white_pawn7',
            'h2': 'white_pawn8',
        }
        
    def get_best_move(self, time_limit=0.1):
        """
        Get the best move from the engine along with metadata.
        time_limit: time in seconds for the engine to think about the move.
        Returns a dictionary with move info.
        """
        result = self.engine.play(self.board, chess.engine.Limit(time=time_limit))
        move = result.move

        from_sq = move.from_square
        to_sq = move.to_square

        moving_piece = self.board.piece_at(from_sq)
        moving_type = PIECE_TYPE_MAP[moving_piece.piece_type]
        
        captured_piece = self.board.piece_at(to_sq) if self.board.is_capture(move) else None
        captured_type = PIECE_TYPE_MAP[captured_piece.piece_type] if captured_piece else None
        
        # Get the piece name from the square mapping
        from_square_name = chess.square_name(from_sq)
        piece_name = self.square_to_piece_name.get(from_square_name)
        
        if piece_name is None:
            # This shouldn't happen in normal play, but handle gracefully
            print(f"Warning: No piece found on square {from_square_name}")
            piece_name = None
        
        move_info = {
            'move': move,
            'from_square': from_square_name,           # "e2"
            'to_square': chess.square_name(to_sq),     # "e4"
            'moving_type': moving_type,                # "pawn"
            'piece_name': piece_name,                  # "white_pawn5" or None
            'is_capture': captured_piece is not None,  # True/False
            'captured_type': captured_type             # "rook" or None
        }

        return move_info

    def apply_move(self, move: chess.Move):
        """
        Apply a move to the internal board state and update the piece mapping.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        from_square_name = chess.square_name(from_sq)
        to_square_name = chess.square_name(to_sq)
        
        # Get the piece name that's moving
        piece_name = self.square_to_piece_name.get(from_square_name)
        
        # Check if this is a capture before applying it
        is_capture = self.board.is_capture(move)
        
        # Apply the move to the chess board
        self.board.push(move)
        
        # Update the piece mapping
        if piece_name is not None:
            # Remove piece from source square
            del self.square_to_piece_name[from_square_name]
            
            # Handle captures: remove captured piece from mapping
            if is_capture:
                # The captured piece was already on the destination square
                if to_square_name in self.square_to_piece_name:
                    del self.square_to_piece_name[to_square_name]
            
            # Place moving piece on destination square
            self.square_to_piece_name[to_square_name] = piece_name

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        """
        return self.board.is_game_over()

    def reset_board(self, fen):
        """
        Reset the board to initial position or a given FEN string.
        """
        if fen is None:
            self.board.reset()
        else:
            self.board.set_fen(fen)

    def close_engine(self):
        """
        Close the chess engine.
        """
        self.engine.quit()


