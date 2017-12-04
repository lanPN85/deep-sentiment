import sys
sys.path.extend(['.'])

from flask import Flask
from flask_cors import CORS
from argparse import ArgumentParser
from sentiment.model import SentimentNet

import flask
import os
import shutil
import webbrowser

import utils

UPLOAD_FOLDER = './uploads'
UPLOAD_NAME = 'temp'

app = Flask(__name__, static_folder='./dist/static',
            template_folder='./dist')


# Serves up webpack Vue app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    return flask.render_template('index.html')


# APIs
__predicted = False
__documents = []
__summary = {}
__MODEL = None
__lock = False


@app.route('/api/upload/', methods=['POST'])
def upload_file():
    global UPLOAD_NAME, __predicted, __lock

    while __lock:
        pass

    file = flask.request.files['file']
    if not utils.allowed_file(file.filename):
        return flask.make_response(flask.jsonify({
            'error': 'File type not allowed. Please upload a file in CSV, TSV or TXT format.'}), 400)

    UPLOAD_NAME = 'temp.' + utils.extension(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, UPLOAD_NAME))

    __predicted = False
    return flask.jsonify({'error': None})


@app.route('/api/summary/', methods=['GET', 'POST'])
def summarize_data():
    global __lock

    if UPLOAD_NAME is None:
        flask.abort(400)

    while __lock:
        pass

    if not __predicted:
        predict()

    return flask.jsonify(__summary)


@app.route('/api/listing/<label>')
def list_documents(label):
    global __lock

    if UPLOAD_NAME is None:
        flask.abort(400)

    while __lock:
        pass

    if not __predicted:
        predict()

    filtered_docs = list(filter(lambda doc: doc['label'] == label, __documents))
    return flask.jsonify(filtered_docs)


def predict():
    global __documents, __summary, __predicted, __lock

    __lock = True

    __documents = []

    docs = utils.extract_docs(os.path.join(UPLOAD_FOLDER, UPLOAD_NAME))
    scores = __MODEL.predict_batch(docs, verbose=1, batch_size=100)

    pos, uns, neg = 0, 0, 0
    for doc, score in zip(docs, scores):
        if score[0] < 0.35:
            label = 'negative'
            neg += 1
        elif score[0] < 0.65:
            label = 'unsure'
            uns += 1
        else:
            label = 'positive'
            pos += 1
        __documents.append({
            'content': doc[:100] + '...', 'score': '%.3f' % float(score[0]),
            'label': label, 'full': doc
        })

    __summary = {
        'positive': pos, 'unsure': uns, 'negative': neg
    }

    __predicted = True
    __lock = False


# Execution
def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', default='../trained/default', dest='MODEL',
                        help='Path to a deep-sentiment model directory. Defaults to ../trained/default')
    parser.add_argument('--port', default=None, dest='PORT',
                        help='The port that the server listens to. Defaults to 5000.')
    parser.add_argument('--debug', dest='DEBUG', action='store_true',
                        help='Toggle debugging mode.')

    return parser.parse_args()


def main(args):
    global __MODEL

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    __MODEL = SentimentNet.load(args.MODEL)

    if args.DEBUG:
        cors = CORS(app, resources={
            r'/api/*': {'origins': '*'}
        })

    webbrowser.open_new_tab('http://localhost:5000')
    try:
        app.run(host='0.0.0.0', port=args.PORT)
    finally:
        print()
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)


if __name__ == '__main__':
    main(parse_arguments())
