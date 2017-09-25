# Model settings
LSTM_LAYERS = (64,)
CNN_LAYERS = (20,)
CNN_FILTERS = (16,)

# Loader settings
DATA_PATH = 'data/movies/'
DOC_LEN = 300
WORDVEC_PATH = 'data/fasttext/imdb.en'
DOC_CUTOFF = None

# Training settings
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.0
STRIDES = 2
BATCH_SIZE = 30
