from argparse import ArgumentParser

import os

from sentiment.model import SentimentNet
from sentiment.loader import SentimentCompactLoader


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='MODEL',
                        help='Path to the pretrained model.')
    parser.add_argument('-s', '--set', dest='SET', default='test',
                        help='The key of the evaluation dataset. Defaults to test.')
    parser.add_argument('-f', '--file', dest='FILE', default=None,
                        help='Path to the formatted TSV file to be used for evaluation. '
                             'Overrides --set.')

    return parser.parse_args()


def main(args):
    try:
        print('Loading model from %s ... ' % args.MODEL)
        model = SentimentNet.load(args.MODEL)
        model.loader.load_data(args.SET)
        print('Done.')

        print('Evaluating...')
        if args.FILE is not None:
            doc_len = model.loader.doc_len
            wv_path = model.loader.wv_path
            tokenizer = model.loader.tokenizer

            data_dir, filename = os.path.split(args.FILE)
            model.loader = SentimentCompactLoader(data_dir, files=[filename], keys=[args.SET],
                                                  wv_path=wv_path, doc_len=doc_len, tokenizer=tokenizer)

        metrics = model.evaluate_generator(test_key=args.SET, batch_size=100)

        print('Done.')

        print()
        for m in metrics:
            print('%s: %s' % (m[0], m[1]))
    except KeyboardInterrupt:
        print('Evaluation interrupted...')


if __name__ == '__main__':
    main(parse_arguments())
