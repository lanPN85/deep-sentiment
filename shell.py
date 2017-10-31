from argparse import ArgumentParser

import sys
import os

from sentiment.model import SentimentNet


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='MODEL',
                        help='Path to the pretrained model.')
    parser.add_argument('-f', dest='FILE', default=None,
                        help='Optional path to a text file containing documents. '
                             'May be a TSV data file (e.g test,tsv).')
    parser.add_argument('-o', dest='OUT', default=sys.stdin,
                        help='Path to output file. Only applies when using -f.')

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

        if args.FILE is not None:
            if args.OUT != sys.stdin:
                with open(args.OUT, 'wt') as outf:
                    return filemain(args.FILE, outf, model)
            return filemain(args.FILE, args.OUT, model)

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


def filemain(fname, outfile, model):
    if outfile != sys.stdin:
        print('Reading from %s...' % fname, file=sys.stderr, end=' ')

    f = open(fname, 'rt')
    contents, labels = [], []
    lines = f.readlines()
    f.close()

    for l in lines:
        cols = l.split('\t', maxsplit=1)
        if len(cols) == 1:
            contents.append(cols[0].strip())
            labels.append('')
        else:
            contents.append(cols[1].strip())
            labels.append(cols[0].strip())

    print('Done', file=sys.stderr)

    length = len(contents)

    print('Score\tPrediction\tLabel\tContent', file=outfile)
    for i, c, l in zip(range(1, length + 1), contents, labels):
        if outfile != sys.stdin:
            print('\t%d/%d' % (i, length), file=sys.stderr, end='\r')

        scores = model.predict(c)
        pred = 'Positive' if scores[0] > 0.5 else 'Negative'
        print('%.3f\t%s\t%s\t%s' % (scores[0], pred, l, c),
              file=outfile)
    print(file=sys.stderr)


if __name__ == '__main__':
    main(parse_arguments())

