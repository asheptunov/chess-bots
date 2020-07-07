import os
import sys
import argparse
import logging
import csv
import time
import ctypes
import multiprocessing
import numpy as np
import torch
from typing import Dict

import chess
import eval_func_convo_net as convonet

def write_snapshots(snapshots, outcomes, turns, snapshot_file, append=False):
    '''
    Writes a list of snapshots to the snapshot_file. Appends to the file if append=True, otherwise truncates
    the file first. For each snapshot, writes a comma separated row with the snapshot (board fen), outcome,
    and the turn number.
    '''
    assert len(snapshots) == len(outcomes) and len(outcomes) == len(turns)
    with open(snapshot_file, 'a' if append else 'w') as snapshot_file_:
        writer = csv.writer(snapshot_file_)
        for i in range(len(turns)):
            writer.writerow([snapshots[i], outcomes[i], turns[i]])

def play_game(white_player, black_player, max_turns:int):
    '''
    Plays a game of chess using the specified models for the white and black players. Stops playing once
    a player wins, the game ends in a draw, or max_turns have been played without either prior outcome.
    Models should be callable using model(board) and should return a legal move to apply to the board.
    Captures snapshots of every board state throughout the game.
    Returns a tuple containing a list of all board snapshots (fens) and the outcome of the game (0 if white won,
    1 if black won, 2 if the game ended in a draw, and -1 if the game didn't end within max_turns turns).
    '''
    # 0. initialize
    board = chess.Board.from_fen(chess.STARTING_FEN)
    snapshots = []  # a list of snapshots (intermediate board states) throughout the game
    counts:Dict[str, int] = {}
    turn = 0  # the current turn (starts with 0)
    player = 0  # 0 = white; 1 = black
    outcome = -1  # the player that won

    # bar = None
    while True:
        # 1. take a snapshot
        fen = board.to_fen()
        snapshots.append(fen)
        counts[fen] = 1 if fen not in counts else counts[fen] + 1
        # 2a. stop if the current player lost
        if board.is_mate():
            outcome = int(not player)
            break
        # 2b. stop if stalemate / insufficient material / fifty-move rule
        if board.is_stalemate():
            outcome = 2
            break
        # 2c. stop if the game is a draw by repitition
        if counts[fen] == 3:
            outcome = 2
            break
        # 2d. stop if we hit the max game length
        if turn >= max_turns:
            break
        # 3. get and apply the model's preferred move
        move:chess.Move = None
        if player == 1:
            move = black_player(board)
        else:
            move = white_player(board)
        board.apply_move(move)
        # 4. update bookkeeping
        turn += 1
        player = not player
    return (snapshots, outcome)

def play_n_games(num_games:int, white_player, black_player, max_turns:int, num_samples_per_game:int, no_unfinished:bool, snapshot_file, snapshot_file_lock=None):
    '''
    Evaluates the given models by forcing them to play n games against each other.
    Appends up to num_samples_per_game distinct snapshots of each game to the snapshot_file, synchronizing on the snapshot_file_lock
    if specified.
    Returns the outcome totals (0 = white won, 1 = black won, 0.5 = draw, -1 = unfinished)
    '''
    if num_games < 1:
        logging.warning('must play at least 1 game but specified %d; defaulting to play 1 game' % num_games)
        num_games = 1
    i = 0
    outcome_totals = {0: 0, 1: 0, 2: 0, -1: 0}
    # play the games
    while i < num_games:
        snapshots, outcome = play_game(white_player, black_player, max_turns)
        # ignore unfinished games if specified
        if outcome == -1 and no_unfinished:
            logging.debug('eval worker %d skipped unfinished game %d as no-unfinished specified' % (os.getpid(), i))
            continue
        if outcome not in outcome_totals:
            logging.warning('eval worker %d skipped game %d due to bad outcome %d' % (os.getpid(), i, outcome))
            continue
        outcome_totals[outcome] += 1
        # sample at most num_samples_per_game distinct snapshots
        turns = np.sort(np.random.choice(len(snapshots), min(num_samples_per_game, len(snapshots)), replace=False))
        snapshots = np.array(snapshots)[turns]
        outcome = np.repeat(outcome, len(turns))
        if snapshot_file_lock is not None:
            snapshot_file_lock.acquire()
        write_snapshots(snapshots, outcome, turns, snapshot_file, append=True)
        if snapshot_file_lock is not None:
            snapshot_file_lock.release()
        # aggregate results
        i += 1
        logging.debug('eval worker %d played %d/%d games (%.2f%%)' % (os.getpid(), i, num_games, 100.0 * i / num_games))
    return outcome_totals

def model_to_player_wrapper(evaluate, board:chess.Board):
    '''
    Wrapper for returning a move to apply from an evaluation function and a board.
    '''
    moves = board.get_moves()
    utilities = np.zeros(len(moves))
    # future_boards = [board.apply_move(move) for move in moves]  # copy-based
    for i in range(len(moves)):
        future_board = chess.Board.from_board(board)
        future_board.apply_move(moves[i])
        future_tensor = convonet.board_tensor(future_board, one_hot=True, unsqueeze=True, half=False)
        utilities[i] = evaluate(future_tensor).item()
    utilities = np.exp(utilities) / np.exp(utilities).sum()  # softmax
    choice = np.random.choice(moves, size=1, p=utilities).item()
    return choice

class ModelEvalWorker(multiprocessing.Process):
    def __init__(self, num_games: int, model_file, max_turns: int, num_samples_per_game: int, no_unfinished: bool, snapshot_file, snapshot_file_lock):
        super(ModelEvalWorker, self).__init__()
        self.num_games = num_games
        self.max_turns = max_turns
        self.num_samples_per_game = num_samples_per_game
        self.no_unfinished = no_unfinished
        self.snapshot_file = snapshot_file
        self.snapshot_file_lock = snapshot_file_lock
        self.model = convonet.EvalFuncConvoNet()
        self.model.load_state_dict(torch.load(model_file))
        self.player = lambda board: model_to_player_wrapper(self.model, board)
        logging.info('eval worker %d loaded model from %s' % (os.getpid(), model_file))

    def run(self):
        logging.info('eval worker %d started' % os.getpid())
        np.random.seed((os.getpid() * hash(time.time())) % (2 ** 32 - 1))  # all workers should produce unique games
        self.model.eval()
        with torch.no_grad():
            outcome_totals = play_n_games(
                self.num_games,
                self.player,
                self.player,
                self.max_turns,
                self.num_samples_per_game,
                self.no_unfinished,
                self.snapshot_file,
                snapshot_file_lock=self.snapshot_file_lock)
        logging.info('eval worker %d finished with outcome totals %s' % (os.getpid(), outcome_totals))

def model_eval(serial:int, load:bool, num_games:int, num_workers:int, num_turns:int, num_samples:int, no_unfinished:bool):
    if serial is None or serial < 0:
        raise ValueError('invalid serial %s for model_eval' % str(serial))
    if load is None:
        load = False
    if num_games is None:
        num_games = 1_000
    if num_workers is None:
        num_workers = 1
    if num_turns is None:
        num_turns = 100
    if num_samples is None:
        num_samples = 10
    if no_unfinished is None:
        no_unfinished = False

    # init logging
    ID = serial
    LOG_DIR = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'log')  # ../log
    LOG_FILE = os.path.join(LOG_DIR, 'eval_%d.log' % ID)
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(message)s')

    # init game parameters
    NUM_GAMES = num_games
    NUM_WORKERS = num_workers
    MAX_TURNS = num_turns
    NUM_SAMPLES_PER_GAME = num_samples
    NO_UNFINISHED = no_unfinished
    logging.info('playing %d games with %d worker(s), %d max turns, %d samples per game, %s unfinished games' % (NUM_GAMES, NUM_WORKERS, MAX_TURNS, NUM_SAMPLES_PER_GAME, 'no' if NO_UNFINISHED else 'including'))

    # init datasets
    DATA_DIR = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data')  # ../data
    SNAPSHOT_FILE = os.path.join(DATA_DIR, 'snapshots_%d.csv' % ID)  # we will make this
    SNAPSHOT_FILE_LOCK = multiprocessing.Lock()
    logging.info('will write dataset to %s' % SNAPSHOT_FILE)

    # init model
    MODEL_FILE = os.path.join(DATA_DIR, 'model_%d.pt' % ID)
    model = convonet.EvalFuncConvoNet()
    if not load:
        # not loading; save the randomly initialized model
        torch.save(model.state_dict(), MODEL_FILE)
        logging.info('load option not specified, randomly initialized and saved model parameters to %s' % MODEL_FILE)

    # init workers
    workers = []
    for i in range(NUM_WORKERS):
        workers.append(ModelEvalWorker(NUM_GAMES, MODEL_FILE, MAX_TURNS, NUM_SAMPLES_PER_GAME, NO_UNFINISHED, SNAPSHOT_FILE, SNAPSHOT_FILE_LOCK))

    # play games to build snapshots dataset
    start_time = time.time()
    if NUM_WORKERS == 1:
        # assume worker role
        logging.info('eval leader assumed worker role as num-workers=1')
        workers[0].run()  # synchronous
    else:
        # start workers
        for worker in workers:
            worker.start()  # asynchronous
        logging.info('eval leader spawned %d worker(s), waiting' % NUM_WORKERS)
        # await workers
        for worker in workers:
            worker.join()  # synchronous
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info('finished in %d h %d m %.1f s' % (elapsed / 3600, elapsed / 60, elapsed % 60))
    logging.info('wrote dataset to %s' % SNAPSHOT_FILE)
    return ID

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model_eval')
    parser.add_argument('-s', '--serial', required=True, dest='serial', type=int, help='serial id of the evaluation run')
    parser.add_argument('-l', '--load', dest='load', action='store_true', help='enables model loading; if enabled, will search for a model in ../data/model_serial.pt; if disabled, will randomly initialize a model and save to ../data/model_serial.pt')
    parser.add_argument('-ng', '--num-games', dest='num_games', type=int, help='number of games per worker to simulate')
    parser.add_argument('-nw', '--num-workers', dest='num_workers', type=int, help='number of worker sub-processes')
    parser.add_argument('-nt', '--num-turns', dest='num_turns', type=int, help='max number of turns per game')
    parser.add_argument('-ns', '--num-samples', dest='num_samples', type=int, help='max number of game state samples to record per game')
    parser.add_argument('-f', '--no-unfinished', dest='no_unfinished', action='store_true', help="disables recording games that didn't finish in num-turns turns; plays num-games finished games")
    args = parser.parse_args()
    serial = model_eval(args.serial, args.load, args.num_games, args.num_workers, args.num_turns, args.num_samples, args.no_unfinished)
    sys.exit(serial)
