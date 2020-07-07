import os
import sys
import argparse
import logging

import model_eval
import model_train

def serial_check(init, current):
    if model_serial < init:
        logging.critial('non-decreasing serial %d w.r.t. initial serial %d; aborting' % (current, init))
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model_auto')
    parser.add_argument('-s', '--serial', type=int, default=1, dest='serial', help='serial number to start from')
    parser.add_argument('-f', '--first', choices=['init', 'eval', 'train'], default='init', dest='first', help='whether to start by randomly initializing a model with the given serial number, by evaluating a pre-initialized model with the given serial number, or by training a pre-initialized and pre-evaluated model with the given serial number')
    parser.add_argument('-n', '--num-iterations', type=int, default=-1, dest='num_iterations', help='number of iterations to perform the train-evaluate loop; specify -1 for unlimited iteration (in this case the script will not terminate)')
    args = parser.parse_args()

    # init serial
    model_serial = INIT_SERIAL = args.serial

    # init logging
    LOG_DIR = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'log')  # ../log
    LOG_FILE = os.path.join(LOG_DIR, 'auto_%d.log' % INIT_SERIAL)
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(message)s')

    # init script args
    src_dir = os.path.dirname(__file__)
    eval_script = os.path.join(src_dir, 'model_eval.py')
    train_script = os.path.join(src_dir, 'model_train.py')
    eval_args = {
        '--serial' : model_serial,
        '--load' : False,
        '--num-games' : 2_000-200,  # TODO: change this
        '--num-workers' : 1,
        '--num-turns' : 200,
        '--num-samples' : 20,
        '--no-unfinished' : True
    }
    train_args = {
        '--serial' : model_serial,
        '--p-dist' : '0.8,0.1,0.1',
        '--epochs' : 10,
        '--batch-size' : 256,
        '--shuffle' : True,
        '--drop-last' : True,
        # use default optimizer hyperparameters (lr, momentum, decay)
        '--learning-rate' : None,
        '--momentum': None,
        '--weight-decay' : None
    }
    logging.info('set up eval script args: %s' % eval_args)
    logging.info('set up train script args: %s' % train_args)

    if args.first == 'init':
        # 1. randomly initialize and evaluate the model
        logging.info('calling model_eval with args %s' % eval_args)
        # model_serial = subprocess.call(eval_args)
        model_serial = model_eval.model_eval(*(eval_args.values()))
        logging.info('initialized model, got serial %d' % model_serial)
    else:
        logging.info('no initializing model_eval as first=%s' % args.first)

    eval_args['--load'] = True  # subsequent calls will load trained models
    logging.info('set --load flag in eval args')

    # 2. for ever:
    i = 0
    N = args.num_iterations
    while N <= -1 or i < N:
        logging.info('iteration %d' % i)

        if i > 0 or args.first != 'eval':  # if first == eval, skip the training step on the first iteration
            # 2a. train the model
            train_args['--serial'] = model_serial
            logging.info('calling model_train with args %s' % train_args)
            # model_serial_new = subprocess.call(train_args)
            model_serial_new = model_train.model_train(*(train_args.values()))
            logging.info('trained model %d, got serial %d' % (model_serial, model_serial_new))
            model_serial = model_serial_new
            serial_check(INIT_SERIAL, model_serial)

        # 2b. evaluate the model
        eval_args['--serial'] = model_serial
        logging.info('calling model_eval with args %s' % eval_args)
        # model_serial_new = subprocess.call(eval_args)
        model_serial_new = model_eval.model_eval(*(eval_args.values()))
        logging.info('evaluated model %d, got serial %d' % (model_serial, model_serial_new))
        model_serial = model_serial_new
        serial_check(INIT_SERIAL, model_serial)

        if N > -1:
            i += 1
