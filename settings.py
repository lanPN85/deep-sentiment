# Model settings
LSTM_LAYERS = (64,)
CNN_LAYERS = (3,)
CNN_FILTERS = (64,)
STRIDES = 1

# Loader settings
DATA_PATH = 'data/amazon/'
DOC_LEN = 400
WORDVEC_PATH = 'data/fasttext/imdb.en'
DOC_CUTOFF = None

# Training settings
LEARNING_RATE = 0.001
EPOCHS = 100
DROPOUT = 0.0
BATCH_SIZE = 30
