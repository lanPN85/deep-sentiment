from keras.callbacks import Callback


class SentimentCallback(Callback):
    def __init__(self, target, save_monitor='val_loss', mode='desc'):
        Callback.__init__(self)
        self._target = target
        self._save_monitor = save_monitor
        self._mode = mode

        if mode == 'desc':
            self._best = float('inf')
        elif mode == 'asc':
            self._best = -float('inf')
        else:
            raise ValueError("Invalid mode '%s'. Mode must be either 'asc' or 'desc'.")
        self._mode_map = {
            'asc': self.is_asc,
            'desc': self.is_desc
        }

    def on_epoch_end(self, epoch, logs=None):
        prev = self._best
        current = logs[self._save_monitor]
        if self._mode_map[self._mode](prev, current):
            self._best = current
            print('\nSaving model to %s...' % self._target.directory, end='')
            self._target.save()
            print('Done.')

    @staticmethod
    def is_asc(prev, current):
        return prev < current

    @staticmethod
    def is_desc(prev, current):
        return prev > current
