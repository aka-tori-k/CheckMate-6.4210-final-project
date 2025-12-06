from chess_interface import ChessInterface

if __name__ == "__main__":
    # Test 5 moves
    chess_interface = ChessInterface()
    for i in range(5):
        meta = chess_interface.get_best_move()
        print(meta)
        
        chess_interface.apply_move(meta["move"])  # update board
    
    chess_interface.close_engine()