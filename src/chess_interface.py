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
        
        move_info = {
            'move': move,
            'from_square': chess.square_name(from_sq), # "e2"
            'to_square': chess.square_name(to_sq),     # "e4"
            'moving_type': moving_type,                # "pawn"
            'is_capture': captured_piece is not None,  # True/False
            'captured_type': captured_type             # "rook" or None
        }

        return move_info

    def apply_move(self, move: chess.Move):
        """
        Apply a move to the internal board state.
        """
        self.board.push(move)

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


