from chess_board import build_chess_board
import time

env = build_chess_board()
print("Meshcat:", env["meshcat"].web_url())

while True:
    time.sleep(1)