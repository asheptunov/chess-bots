import os
import sys
import argparse
import logging
import csv
import time
import copy
import numpy as np
import torch
import torch.utils.data

import chess
import eval_func_convo_net as convonet

class ChessDataset(torch.utils.data.Dataset):
    '''
    Dataloader for chess snapshot datasets
    '''
    def __init__(self, snapshot_file):
        self.boards = []
        self.labels = []
        with open(snapshot_file, 'r') as snapshot_file_:
            reader = csv.reader(snapshot_file_)
            for row in reader:
                # ignore turn counter for now
                # self.boards.append(convonet.board_tensor(chess.board_make(row[0]), one_hot=True, unsqueeze=False, half=False))
                self.boards.append(convonet.board_tensor(chess.Board.from_fen(row[0]), one_hot=True, unsqueeze=False, half=False))
                label = float(row[1])
                self.labels.append(torch.tensor([0.5 if label == 2 else label]))
        self.boards = torch.stack(self.boards)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.boards[idx], self.labels[idx])

def split_dataset(dataset_file, p_dist=(0.8, 0.2), shuffle=True):
    '''
    Splits a dataset at the specified file by rows (shuffling first if specified) and according
    to the specified probability distribution.
    Returns a list of the filenames of the generated dataset shards, ordered corresponding to p_dist
    by cardinality relative to the original dataset.
    '''
    rows = None
    with open(dataset_file, 'r') as dataset_file_:
        reader = csv.reader(dataset_file_)
        rows = np.array([row for row in reader])
    if shuffle:
        np.random.shuffle(rows)
    p_dist = (len(rows) * np.cumsum([0] + list(p_dist))).astype(int)
    shards = [rows[p_dist[i]:p_dist[i+1],:] for i in range(len(p_dist) - 1)]
    shard_files = ['%s_%d.csv' % (dataset_file[:-4], i) for i in range(len(shards))]
    for i in range(len(shards)):
        with open(shard_files[i], 'w') as shard_file:
            writer = csv.writer(shard_file)
            for row in shards[i]:
                writer.writerow(row)
    cardinalities = [len(shard) for shard in shards]
    return (shard_files, cardinalities)

def train(model, train_loader, optim):
    model.train()
    losses = []
    for i, (data, label) in enumerate(train_loader):
        optim.zero_grad()
        prediction = model(data)
        loss = model.loss(prediction, label)
        loss.backward()
        losses.append(loss)
        optim.step()
        logging.debug('batch iteration %d loss %.3f' % (i, loss.item()))
    mean_loss = torch.mean(torch.tensor(losses)).item()
    return mean_loss

def test(model, test_loader):
    model.eval()
    losses = []
    correct = []
    with torch.no_grad():
        for (data, label) in test_loader:
            prediction = model(data)
            loss = model.loss(prediction, label).item()
            losses.append(loss)
            correct.append(torch.eq(torch.round(prediction), label))
    mean_loss = torch.mean(torch.tensor(losses)).item()
    mean_acc = torch.mean(torch.flatten(torch.cat(correct)).double()).item()
    return (mean_loss, mean_acc)

def model_train(serial:int, pdist, epochs:int, batch_size:int, shuffle:bool, drop_last:bool, learning_rate:float, momentum:float, weight_decay:float):
    if serial is None or serial < 0:
        raise ValueError('invalid serial %s for model_train' % str(serial))
    if pdist is None:
        pdist = '0.8,0.1,0.1'
    if epochs is None:
        epochs = 20
    if batch_size is None:
        batch_size = 64
    if shuffle is None:
        shuffle = True
    if drop_last is None:
        drop_last = True
    if learning_rate is None:
        learning_rate = 1e-3
    if momentum is None:
        momentum = 0.9
    if weight_decay is None:
        weight_decay = 1e-5

    # init logging
    ID = serial
    LOG_DIR = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'log')  # ../log
    LOG_FILE = os.path.join(LOG_DIR, 'train_%d.log' % ID)
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(message)s')

    # init hyperparameters
    P_DIST = [float(p) for p in pdist.split(',')]
    if len(P_DIST) != 3 or sum(P_DIST) - 1 > 1e-4:
        P_DIST = [0.8, 0.1, 0.1]
        logging.warning('bad pdist %s; defaulting to %s' % (pdist, P_DIST))
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    SHUFFLE = shuffle
    DROP_LAST = drop_last
    LEARNING_RATE = learning_rate
    MOMENTUM = momentum
    WEIGHT_DECAY = weight_decay

    # split dataset
    DATA_DIR = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data')  # ../data
    DATASET_FILE = os.path.join(DATA_DIR, 'snapshots_%d.csv' % ID)
    logging.info('splitting dataset %s by p-dist %s' % (DATASET_FILE, P_DIST))
    shard_files, shard_cardinalities = split_dataset(DATASET_FILE, p_dist=P_DIST)
    train_file, val_file, test_file = shard_files
    train_n, val_n, test_n = shard_cardinalities
    logging.info('training size %d, validation size %d, test size %d' % (train_n, val_n, test_n))
    logging.info('training file %s, validation file %s, test file %s' % (train_file, val_file, test_file))
    if DROP_LAST and BATCH_SIZE > min(shard_cardinalities):
        DROP_LAST = False
        logging.warning('drop-last specified but batch size %d exceeds minimum dataset size %d; defaulting to not drop-last' % (BATCH_SIZE, min(shard_cardinalities)))

    # init datasets
    train_loader = torch.utils.data.DataLoader(ChessDataset(train_file), batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=DROP_LAST)
    val_loader = torch.utils.data.DataLoader(ChessDataset(val_file), batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=DROP_LAST)
    test_loader = torch.utils.data.DataLoader(ChessDataset(test_file), batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=DROP_LAST)

    # init model
    MODEL_FILE = os.path.join(DATA_DIR, 'model_%d.pt' % ID)
    model = convonet.EvalFuncConvoNet()
    model.load_state_dict(torch.load(MODEL_FILE))
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    logging.info('loaded model parameters from %s' % MODEL_FILE)

    # train model
    logging.info('training for %d epochs, batch size %d, %s dropping last batch, %s shuffling each epoch, learning rate %.5f, momentum %.5f, decay %.5f' \
        % (EPOCHS, BATCH_SIZE, '' if DROP_LAST else 'not', '' if SHUFFLE else 'not', LEARNING_RATE, MOMENTUM, WEIGHT_DECAY))
    start_time = time.time()
    best_val_loss = np.inf
    best_i = 0
    best_model = None
    for i in range(-1, EPOCHS):
        # train the model for an epoch
        mean_train_loss = train(model, train_loader, optimizer) if i > -1 else np.inf
        mean_val_loss, val_acc = test(model, val_loader)
        logging.info('epoch %d, training loss %.5f, validation loss %.5f, validation acc %.5f' % (i, mean_train_loss, mean_val_loss, val_acc))
        # save best model
        if mean_val_loss < best_val_loss and i > -1:
            best_val_loss = mean_val_loss
            best_i = i
            best_model = copy.deepcopy(model)
            logging.info('new best model with validation loss %.5f' % best_val_loss)
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info('finished in %d h %d m %.1f s' % (elapsed / 3600, elapsed / 60, elapsed % 60))

    # test model
    if best_model is not None:
        # test best model
        mean_test_loss, test_acc = test(best_model, test_loader)
        logging.info('best model test loss %.5f, test acc %.5f' % (mean_test_loss, test_acc))
        # save best model
        best_model_file = os.path.join(DATA_DIR, 'model_%d.pt' % (ID + best_i + 1))
        torch.save(best_model.state_dict(), best_model_file)
        logging.info('best model saved to %s' % best_model_file)
    else:
        logging.warning("model validation loss didn't improve at any epoch beyond base model")

    # return the id of the best trained model to eval on
    return ID + best_i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model_train')
    parser.add_argument('-s', '--serial', required=True, dest='serial', type=int, help='serial id of the evaluation run')
    parser.add_argument('-p', '--p-dist', dest='pdist', help='probability distribution to split dataset into for training-validation-testing; expects three (3) comma-separated floating-point values')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, help='number of epochs to train for')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, help='training batch size')
    parser.add_argument('-sh', '--shuffle', dest='shuffle', action='store_true', help='enables shuffling datasets on each epoch')
    parser.add_argument('-dl', '--drop-last', dest='drop_last', action='store_true', help='enables dropping the last (possibly incomplete) batch per epoch')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, help='learning rate hyperparameter for training')
    parser.add_argument('-m', '--momentum', dest='momentum', type=float, help='momentum hyperparameter for training')
    parser.add_argument('-wd', '--weight-decay', dest='weight_decay', type=float, help='weight decay hyperparameter for training')
    args = parser.parse_args()
    serial = model_train(args.serial, args.pdist, args.epochs, args.batch_size, args.shuffle, args.drop_last, args.learning_rate, args.momentum, args.weight_decay)
    sys.exit(serial)
