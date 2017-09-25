from argparse import ArgumentParser

import os

from sentiment.settings import *
from sentiment.model import SentimentNet
from sentiment.loader import SentimentDataLoader


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--name', default='default', dest='NAME',
                        help='The name for the directory containing the trained model.'
                             'This directory will always be inside \'trained/\'')
    parser.add_argument('-lr', '--learning-rate', type=float, default=LEARNING_RATE, dest='LR',
                        help="The model's initital learning rate")
    parser.add_argument('--doc-len', default=DOC_LEN, dest='DLEN', type=int,
                        help="The maximum length in words for each document")
    parser.add_argument('-bs', '--batch-size', default=BATCH_SIZE, dest='BATCH', type=int,
                        help="The batch size used for training")
    parser.add_argument('--cutoff', default=DOC_CUTOFF, dest='CUT',
                        help="The maximum number of documents to load for each set")
    parser.add_argument('--epochs', default=EPOCHS, dest='EPOCHS', type=int,
                        help="The number of epochs to train the model. "
                             "May stop sooner due to early stop condition.")
    parser.add_argument('-wv', '--wordvec-path', default=WORDVEC_PATH, dest='WV_PATH',
                        help="The path to the fasttext model to be used. "
                             "Extensions like .bin and .vec should be excluded.")
    parser.add_argument('--dropout', default=DROPOUT, type=float, dest='DROPOUT',
                        help='Dropout value to be used across all layers during training.')
    parser.add_argument('--strides', default=STRIDES, type=int, dest='STRIDES',
                        help='Stride value to be used for all CNN layers.')
    parser.add_argument('--data-path', default=DATA_PATH, dest='PATH',
                        help='The directory that contains training and test data. '
                             'See data/README.md for details.')
    parser.add_argument('--resume', action='store_true', dest='RESUME',
                        help='Loads a trained model with name specified by --name and resume training.')
    parser.add_argument('--start-from', default=0, type=int, dest='FROM',
                        help='The epoch to start training from. Useful when resuming.')

    return parser.parse_args()


def main(args):
    os.makedirs('trained', exist_ok=True)
    os.makedirs('trained/' + args.NAME, exist_ok=True)

    if args.RESUME:

        return

    print('Loading data... ', end='')
    loader = SentimentDataLoader(args.PATH, cutoff=args.CUT, doc_len=args.DLEN, wv_path=args.WV_PATH)
    print('Done.')

    print('Creating model...')
    model = SentimentNet(loader, lstm_layers=LSTM_LAYERS, cnn_layers=CNN_LAYERS, cnn_filters=CNN_FILTERS,
                         dropout=args.DROPOUT, strides=args.STRIDES)

    print('Compiling... ', end='')
    model.compile(learning_rate=args.LR)
    print('Done.')

    try:
        print('Starting training...')
        history = model.train(batch_size=args.BATCH, start_from=args.FROM, epochs=args.EPOCHS)
    except KeyboardInterrupt:
        print('\nTraining interrupted...')


if __name__ == '__main__':
    main(parse_arguments())
