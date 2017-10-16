from argparse import ArgumentParser

from sentiment.model import SentimentNet


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='MODEL',
                        help='Path to the pretrained model.')
    parser.add_argument('-s', '--set', dest='SET', default='test',
                        help='The key of the evaluation dataset. Defaults to test.')

    return parser.parse_args()


def main(args):
    print('Loading model from %s ... ' % args.MODEL)
    model = SentimentNet.load(args.MODEL)
    model.loader.load_data(args.SET)
    print('Done.')

    print('Evaluating...')
    metrics = model.evaluate_generator(test_key=args.SET, batch_size=100)
    print('Done.')

    print()
    for m in metrics:
        print('%s: %s' % (m[0], m[1]))


if __name__ == '__main__':
    main(parse_arguments())
