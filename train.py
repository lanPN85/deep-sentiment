from argparse import ArgumentParser

import os

from sentiment.settings import *


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--name', default='default', dest='NAME',
                        help='The name for the directory containing the trained model.'
                             'This directory will always be inside \'trained/\'')
    parser.add_argument('-lr', '--learning-rate', default=LEARNING_RATE, dest='LR',
                        help="The model's initital learning rate")
    parser.add_argument('--doc-len', default=DOC_LEN, dest='DLEN', type=int,
                        help="The maximum length in words for each document")
    parser.add_argument('--cutoff', default=DOC_CUTOFF, dest='CUT',
                        help="The maximum number of documents to load for each set")
    parser.add_argument('--epochs', default=EPOCHS, dest='EPOCHS', type=int,
                        help="The number of epochs to train the model. "
                             "May stop sooner due to early stop condition.")
    parser.add_argument('-wv', '--wordvec-path', default=WORDVEC_PATH, dest='WV_PATH',
                        help="The path to the fasttext model to be used. "
                             "Extensions like .bin and .vec should be excluded.")

    return parser.parse_args()


def main(args):
    os.makedirs('trained', exist_ok=True)
    os.makedirs('trained/' + args.NAME, exist_ok=True)


if __name__ == '__main__':
    main(parse_arguments())
