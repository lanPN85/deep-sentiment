from keras.optimizers import RMSprop
from keras.models import Model

import math


class SentimentNet:
    def __init__(self, loader, lstm_layers=(64,), cnn_layers=(128,), cnn_filters=(128,)):
        self.loader = loader
        self.loader.load_data()

        self._lstm_layers = lstm_layers
        self._cnn_layers = cnn_layers
        self._cnn_filters = cnn_filters
        self._model = self._create_model()

    def _create_model(self):

        model = Model()
        return model

    def compile(self, optimizer=RMSprop, learning_rate=0.001):
        self._model.compile(optimizer=optimizer(lr=learning_rate), loss='categorical_crossentropy',
                            metrics=['acc'])

    def train(self, train_key='train', val_key='val', batch_size=30, start_from=0):
        train_steps = int(math.ceil(self.loader.data_len(train_key) / batch_size))
        val_steps = int(math.ceil(self.loader.data_len(val_key) / batch_size))

        callbacks = []

        self._model.fit_generator(self.loader.generate_data(key=train_key, batch_size=batch_size),
                                  steps_per_epoch=train_steps, max_queue_size=2, initial_epoch=start_from,
                                  validation_data=self.loader.generate_data(key=val_key, batch_size=batch_size),
                                  validation_steps=val_steps, callbacks=callbacks)

    def predict(self, document):
        pass

    def predict_batch(self, documents):
        pass

    def evaluate(self, test_key='test'):
        pass

    def __getitem__(self, item):
        return self.predict(item)
