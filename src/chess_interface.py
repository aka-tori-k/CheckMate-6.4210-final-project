import chess
import chess.engine

# ------- Engine + board setup -------

ENGINE_PATH = "/usr/games/stockfish"  # in the dev container

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
board = chess.Board()

PIECE_TYPE_MAP = {
    chess.PAWN: 'pawn',
    chess.ROOK: 'rook',
    chess.KNIGHT: 'knight',
    chess.BISHOP: 'bishop',
    chess.QUEEN: 'queen',
    chess.KING: 'king'
}

def get_best_move(time_limit=0.1):
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    move = result.move

    from_sq = move.from_square
    to_sq = move.to_square

    moving_piece = board.piece_at(from_sq)
    moving_type = PIECE_TYPE_MAP[moving_piece.piece_type]
    
    captured_piece = board.piece_at(to_sq) if board.is_capture(move) else None
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

def apply_move(move: chess.Move):
    board.push(move)

def is_game_over() -> bool:
    return board.is_game_over()

def close_engine():
    engine.quit()


