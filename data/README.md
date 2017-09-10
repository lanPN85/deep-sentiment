This directory contains all data used for training, evaluating and deploying the models.

## Data Format
Any datasets used for training must adhere to the following format:
- Each dataset must be within its own directory within `data/`.
- The data loader will only load 3 files for each dataset: `train.tsv`, `val.tsv` and `test.tsv`.
- Each of the above file must contain 2 tab-seperated rows. The first contains sentence labels, which should be either `Positive` or `Negative`. These labels will be used for displaying results. The second row contains the data sentences, one example per line corresponding to the labels.

## Word Vectors
- The models use Facebook's fastText word embedding. The embeddings are stored in the `data/fasttext` directory. Make sure to include both the `.vec` and `.bin` files if you use a custom model.
