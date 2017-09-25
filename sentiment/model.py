from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import LSTM, Conv1D, Dense, Masking, Concatenate, Input, Flatten

import math


class SentimentNet:
    def __init__(self, loader, lstm_layers=(64,), cnn_layers=(128,), cnn_filters=(128,),
                 dropout=0.0, strides=1):
        self.loader = loader
        self.loader.load_data()

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
        self._model = self._create_model()

    def _create_model(self):
        # Create LSTM branch
        lstm_branch = Sequential(name='LBranch')
        lstm_branch.add(Masking(input_shape=(self.loader.doc_len, self.loader.embed_dims)))
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
                                  strides=self._strides, activation='relu',
                                  name='CNN_%d' % len(self._cnn_filters)))
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
        train_steps = int(math.ceil(self.loader.data_len(train_key) / batch_size))
        val_steps = int(math.ceil(self.loader.data_len(val_key) / batch_size))

        callbacks = []

        return self._model.fit_generator(self.loader.generate_data(key=train_key, batch_size=batch_size),
                                         steps_per_epoch=train_steps, max_queue_size=2, initial_epoch=start_from,
                                         validation_data=self.loader.generate_data(key=val_key, batch_size=batch_size),
                                         validation_steps=val_steps, callbacks=callbacks, epochs=epochs)

    def predict(self, document):
        pass

    def predict_batch(self, documents):
        pass

    def evaluate(self, test_key='test'):
        pass

    def __getitem__(self, item):
        return self.predict(item)

    @property
    def dropout(self):
        return self._dropout
