from argparse import ArgumentParser

import sys
import os

from sentiment.model import SentimentNet


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='MODEL',
                        help='Path to the pretrained model.')
    return parser.parse_args()


def main(args):
    try:
        print('Loading model from %s ... ' % args.MODEL)
        model = SentimentNet.load(args.MODEL)
        print('Done.')

        if sys.platform.startswith('win'):
            os.system('cls')
        else:
            os.system('clear')

        print('\nDeep Sentiment Shell')
        print('by Phan Ngoc Lan')
        print('Use the -h argument for help. Press Ctrl+C or Ctrl+D to exit.\n')
        while True:
            query = input('>>> ')
            scores = model.predict(query)
            pred = 'Positive' if scores[0] > 0.5 else 'Negative'
            print('\tSentiment score: %.3f' % scores[0])
            print('\tPredicted sentiment: %s' % pred)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == '__main__':
    main(parse_arguments())

