This directory contains all data used for training, evaluating and deploying the models.

## Data Format
Any datasets used for training must adhere to the following format:
- Each dataset must be within its own directory within `data/`.
- The data loader will only load 3 files for each dataset: `train.tsv`, `val.tsv` and `test.tsv`.
- Each of the above file must contain 2 tab-seperated rows. The first contains sentence labels, which should be either `Positive` or `Negative`. These labels will be used for displaying results. The second row contains the data sentences, one example per line corresponding to the labels.

## Word Vectors
- This project uses Facebook's fastText for word embedding. The embeddings should be stored in the `data/fasttext` directory. Make sure to include both the `.vec` and `.bin` files.
- Public fasttext models trained on Wikipedia crawl are available [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). Note that these models can be quite large and may take several minutes to load.

## Download Links
Download and extract the following into their respective folders.
- [Amazon reviews](https://www.kaggle.com/bittlingmayer/amazonreviews): `data/amazon`.
- [IMDB reviews](http://ai.stanford.edu/~amaas/data/sentiment/): `data/movies`.

*Note: Run `python transform.py` after extraction to get the split training files.*
