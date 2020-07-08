import os
import sys
import torch

import chess

import chess
import eval_func_convo_net as convonet

def usage():
    print("usage: model_test.py (fen|'start')")
    sys.exit(1)

if __name__ == '__main__':
    data_dir = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data')  # ../data
    model_file = os.path.join(data_dir, 'model_1.pt')
    model = convonet.EvalFuncConvoNet()
    model.load_state_dict(torch.load(model_file))
    print('loaded model from %s' % model_file)

    print('cuda is %s available' % ('' if torch.cuda.is_available() else 'not'))

    if len(sys.argv) != 2:
        usage()
    fen = chess.STARTING_FEN if sys.argv[1].lower() == 'start' else sys.argv[1]
    with torch.no_grad():
        board = chess.Board.from_fen(fen)
        board_tensor = convonet.board_tensor(board, one_hot=True, unsqueeze=True, half=False)
        pred_util = model(board_tensor).item()
        print('utility %.5f' % pred_util)
