import os
import ctypes

import chess

WPAWN = 0
WKNIGHT = 1
WBISHOP = 2
WROOK = 3
WQUEEN = 4
WKING = 5
BPAWN = 6 + WPAWN
BKNIGHT = 6 + WKNIGHT
BBISHOP = 6 + WBISHOP
BROOK = 6 + WROOK
BQUEEN = 6 + WQUEEN
BKING = 6 + WKING

INFTY = 100_000

def evaluate(board:chess.Board):
  '''
  returns a value for the board based on a simple evaluation function
  '''
  ret = 0
  player = not not ((board.board().contents.flags) & 0b10000)  # True if black, False if white
  for rk in range(0, 8):
    for offs in range(0, 8):
      pc = (board.board().contents.ranks[rk] >> (offs << 2)) & 0xf
      # if (pc / 6) == player:
      if (pc >= BPAWN and pc <= BKING) == player:
        ret += pcval(pc)
      else:
        ret -= pcval(pc)
  return ret

def pcval(pc:int):
  '''
  returns the approximate value of a piece
  '''
  if pc > WKING:
    pc -= 6
  if pc == WPAWN:
    return 1
  if pc == WKNIGHT or pc == WBISHOP:
    return 3
  if pc == WROOK:
    return 5
  if pc == WQUEEN:
    return 9
  return 0

def minimax(evaluate, board:chess.Board, depth:int):
  '''
  performs minimax search from the current board position, exploring down to a given depth;
  returns the optimal move from the current position (None if there are no moves, or depth is 0) and the value of the board when applying that move
  '''
  if depth <= 0:
    return (None, evaluate(board))

  moves = board.get_moves()
  if len(moves) == 0:
    # active player has no moves; loss or stalemate
    return (None, -INFTY - depth)

  max_utility_move = None
  max_utility = -float('inf')
  for move in moves:
    future_board = chess.Board.from_board(board)
    future_board.apply_move(move)
    opp_best_move, opp_utility = minimax(evaluate, future_board, depth - 1)
    utility = -opp_utility  # negate since this is the opponent's maximal utility, which is our minimal utility
    if (utility > max_utility):
      max_utility = utility
      max_utility_move = move

  return (max_utility_move, max_utility)

if __name__ == '__main__':
    board = chess.Board.from_fen(chess.STARTING_FEN)
    depth = 3
    move, utility = minimax(evaluate, board, depth)
    print(board.to_tui())
    print('searched to depth %d' % depth)
    print('best move %s' % str(move))
    print('utility %d' % utility)
