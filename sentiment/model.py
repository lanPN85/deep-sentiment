from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import LSTM, Conv1D, Dense, Masking, Concatenate, Input, Flatten, AveragePooling1D
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger

import keras.backend as K
import math
import os
import numpy as np
import pickle

from sentiment.loader import SentimentCompactLoader
from sentiment.callbacks import SentimentCallback
from sentiment import metrics


class SentimentNet:
    def __init__(self, loader, lstm_layers=(64,), cnn_layers=(128,), cnn_filters=(128,),
                 dropout=0.0, strides=1, directory='./model', weights=None):
        self.loader = loader

        self._lstm_layers = lstm_layers
        self._cnn_layers = cnn_layers
        self._cnn_filters = cnn_filters
        if len(cnn_layers) != len(cnn_filters):
            raise ValueError('The number of CNN kernels must equal the number of filter sets. '
                             'Instead got %d kernels and %d filter sets.' % (len(cnn_layers), len(cnn_filters)))
        for l in cnn_layers:
            if l > self.loader.doc_len:
                raise ValueError('Kernel size %d greater than document size of %d' % (l, self.loader.doc_len))

        self._dropout = dropout
        self._strides = strides
        self._dir = directory
        self._model = self._create_model()
        if weights is not None:
            self._model.load_weights(weights)

    def _create_model(self):
        # Create LSTM branch
        lstm_branch = Sequential(name='LBranch')
        lstm_branch.add(Masking(input_shape=(self.loader.doc_len, self.loader.embed_dims,),
                                name='Masking'))
        for i, n in enumerate(self._lstm_layers[:-1]):
            lstm_branch.add(LSTM(n, activation='tanh', recurrent_activation='hard_sigmoid',
                                 recurrent_dropout=self._dropout, dropout=self._dropout,
                                 return_sequences=True, name='LSTM_%d' % (i + 1)))
        lstm_branch.add(LSTM(self._lstm_layers[-1], activation='tanh', recurrent_activation='hard_sigmoid',
                             recurrent_dropout=self._dropout, dropout=self._dropout,
                             return_sequences=False, name='LSTM_%d' % len(self._lstm_layers)))
        lstm_branch.summary()

        # Create CNN branch
        cnn_branch = Sequential(name='CBranch')
        if len(self._cnn_layers) == 1:
            cnn_branch.add(Conv1D(self._cnn_filters[0], self._cnn_layers[0],
                                  strides=self._strides, activation='relu', name='CNN_1',
                                  input_shape=(self.loader.doc_len, self.loader.embed_dims)))
        else:
            cnn_branch.add(Conv1D(self._cnn_filters[0], self._cnn_layers[0],
                                  strides=self._strides, activation='hard_sigmoid', name='CNN_1',
                                  input_shape=(self.loader.doc_len, self.loader.embed_dims)))
            for i, n in enumerate(self._cnn_layers[1:-1]):
                cnn_branch.add(Conv1D(self._cnn_filters[i + 1], n,
                                      strides=self._strides, activation='hard_sigmoid',
                                      name='CNN_%d' % (i + 2)))
            cnn_branch.add(Conv1D(self._cnn_filters[0], self._cnn_layers[0],
                                  strides=self._strides, activation='tanh',
                                  name='CNN_%d' % len(self._cnn_filters)))
        pooling_size = (int(cnn_branch.output_shape[1]),)
        cnn_branch.add(AveragePooling1D(pool_size=pooling_size, name='Pooling'))
        cnn_branch.add(Flatten(name='Flattener'))
        cnn_branch.summary()

        inp = Input(shape=(self.loader.doc_len, self.loader.embed_dims),
                    name='Vector_Input')
        lstm_out = lstm_branch(inp)
        cnn_out = cnn_branch(inp)

        merger = Concatenate(name='Concat_Merge')
        merged = merger([lstm_out, cnn_out])
        output = Dense(2, activation='softmax',
                       name='Output_Dense')(merged)

        model = Model(inputs=(inp,), outputs=(output,))
        model.summary()
        return model

    def compile(self, optimizer=RMSprop, learning_rate=0.001):
        self._model.compile(optimizer=optimizer(lr=learning_rate), loss='categorical_crossentropy',
                            metrics=['acc'])

    def train(self, train_key='train', val_key='val', batch_size=30, start_from=0, epochs=50):
        self.loader.load_data(key=train_key)
        self.loader.load_data(key=val_key)

        train_steps = int(math.ceil(self.loader.data_len(train_key) / batch_size))
        val_steps = int(math.ceil(self.loader.data_len(val_key) / batch_size))

        cb1 = SentimentCallback(self, save_monitor='val_loss', mode='desc')
        cb2 = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
        cb3 = CSVLogger(os.path.join(self._dir, 'epochs.csv'), append=(start_from > 0))
        cb4 = EarlyStopping(monitor='loss', patience=1, verbose=1)
        callbacks = [cb1, cb2, cb3, cb4]
        if K.backend() == 'tensorflow':
            # noinspection PyTypeChecker
            callbacks.append(TensorBoard(os.path.join(self._dir, 'tensorboard'),
                                         batch_size=batch_size, histogram_freq=1,
                                         write_grads=False, write_graph=True,
                                         write_images=True))

        return self._model.fit_generator(self.loader.generate_data(key=train_key, batch_size=batch_size),
                                         steps_per_epoch=train_steps, max_queue_size=1, initial_epoch=start_from,
                                         validation_data=self.loader.generate_data(key=val_key, batch_size=batch_size),
                                         validation_steps=val_steps, callbacks=callbacks, epochs=epochs)

    def predict(self, document):
        inp = self.loader.doc2mat(document)
        out = self._model.predict(inp, batch_size=1, verbose=0)
        return out[0]

    def predict_batch(self, documents, verbose=1, batch_size=20):
        inp = [self.loader.doc2mat(documents[0])]
        length = len(documents)

        for i, d in enumerate(documents[1:]):
            inp.append(self.loader.doc2mat(d))
            if verbose > 0:
                print('Vectorizing %d/%d' % (i+2, length), end='\r')
        inp = np.asarray(inp, dtype=np.float32)
        inp = np.reshape(inp, (length, self.loader.doc_len, self.loader.embed_dims))
        if verbose > 0:
            print('\nVectorization complete.')
        return self._model.predict(inp, batch_size=batch_size, verbose=verbose)

    def evaluate(self, test_key='test', verbose=1, batch_size=20):
        self.loader.load_data(test_key)
        raw, true_labels = self.loader[test_key]
        preds = self.predict_batch(raw, verbose=verbose, batch_size=batch_size)

        pred_labels = ['Positive' if p[0] > p[1] else 'Negative' for p in preds]

        mets = []
        acc = metrics.accuracy(true_labels, pred_labels)
        mets.append(('Accuracy', acc))

        pos_precision = metrics.precision(true_labels, pred_labels, 'Positive')
        mets.append(('Precision @ `Positive`', pos_precision))
        pos_recall = metrics.recall(true_labels, pred_labels, 'Positive')
        mets.append(('Recall @ `Positive`', pos_recall))
        mets.append(('F1 @ `Positive`', metrics.f1_score(pos_precision, pos_recall)))

        neg_precision = metrics.precision(true_labels, pred_labels, 'Negative')
        mets.append(('Precision @ `Negative`', neg_precision))
        neg_recall = metrics.recall(true_labels, pred_labels, 'Negative')
        mets.append(('Recall @ `Negative`', neg_recall))
        mets.append(('F1 @ `Negative`', metrics.f1_score(neg_precision, neg_recall)))

        return mets

    def save(self):
        f1 = open(os.path.join(self._dir, 'config.pkl'), 'wb')
        wpath = os.path.join(self._dir, 'weights.hdf5')
        lpath = os.path.join(self._dir, 'loader.pkl')

        self.loader.save(lpath)
        self._model.save_weights(wpath)
        config = {
            'lstm_layers': self._lstm_layers,
            'cnn_layers': self._cnn_layers,
            'cnn_filters': self._cnn_filters,
            'dropout': self._dropout,
            'strides': self._strides,
        }
        pickle.dump(config, f1, pickle.HIGHEST_PROTOCOL)
        f1.close()

    @classmethod
    def load(cls, directory):
        f1 = open(os.path.join(directory, 'config.pkl'), 'rb')
        wpath = os.path.join(directory, 'weights.hdf5')
        lpath = os.path.join(directory, 'loader.pkl')

        loader = SentimentCompactLoader.load(lpath)
        config = pickle.load(f1)
        lstm_layers = config['lstm_layers']
        cnn_layers = config['cnn_layers']
        cnn_filters = config['cnn_filters']
        dropout = config.get('dropout', 0.0)
        strides = config['strides']

        model = SentimentNet(loader, lstm_layers=lstm_layers, cnn_layers=cnn_layers,
                             cnn_filters=cnn_filters, dropout=dropout, strides=strides,
                             directory=directory, weights=wpath)
        f1.close()
        return model

    def __getitem__(self, item):
        return self.predict(item)

    @property
    def dropout(self):
        return self._dropout

    @property
    def directory(self):
        return self._dir

    @directory.setter
    def directory(self, value):
        self._dir = value
