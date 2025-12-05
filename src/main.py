from chess_interface import get_best_move, apply_move, is_game_over, close_engine

if __name__ == "__main__":
    # Test 5 moves
    for i in range(5):
        meta = get_best_move()
        print(meta)
        
        apply_move(meta["move"])  # update board
    
    close_engine()