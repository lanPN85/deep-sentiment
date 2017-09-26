from gensim.models.wrappers import FastText

import numpy as np
import nltk
import os
import pickle


class SentimentDataLoader:
    def __init__(self, path, files=('train.tsv', 'val.tsv', 'test.tsv'), keys=('train', 'val', 'test'), cutoff=None,
                 doc_len=100, wv_path='data/fasttext/imdb.en', tokenizer=nltk.word_tokenize):
        self._path = path
        self._cutoff = cutoff
        self._raw = {k: None for k in keys}
        self._labels = {k: None for k in keys}
        self._files = {k: os.path.join(path, f) for k, f in zip(keys, files)}
        self._doc_len = doc_len
        self._wv = self._load_wv(wv_path)
        self._tokenizer = tokenizer
        self.FALLBACK_VECTOR = np.ones((self.embed_dims,), dtype=np.float32)

    @staticmethod
    def _load_wv(path):
        model = FastText.load_fasttext_format(path)
        return model

    def load_data(self, key=None):
        if key is not None:
            self._raw[key], self._labels[key] = self._read_file(self._files[key])
        else:
            for k in self._raw.keys():
                self._raw[k], self._labels[k] = self._read_file(self._files[k])

        if self._cutoff is not None:
            for k in self._raw.keys():
                self._raw[k], self._labels[k] = self._raw[k][:self._cutoff], self._labels[k][:self._cutoff]

    def generate_data(self, key, batch_size):
        l = len(self._raw[key])
        while True:
            for i in range(batch_size, l, batch_size):
                raw = self._raw[key][i-batch_size:i]
                labels = self._labels[key][i-batch_size:i]

                mat = np.zeros((batch_size, self._doc_len, self.embed_dims))
                for i, sent in enumerate(raw):
                    words = self._tokenizer(sent)[:self.doc_len]
                    for j, w in enumerate(words):
                        mat[i][j] = self._get_wordvec(w)

                nlabels = [[0, 1] if label == 'Negative'
                           else [1, 0] for label in labels]
                yield mat, nlabels

    def data_len(self, key):
        return len(self._raw[key])

    def _get_wordvec(self, word):
        try:
            return self._wv[word]
        except KeyError:
            return self.FALLBACK_VECTOR

    @staticmethod
    def _read_file(path):
        f = open(path, 'rt')
        raw, labels = [], []
        for line in f:
            label, content = line.split('\t')
            raw.append(content)
            labels.append(label)
        f.close()
        return raw, labels

    @property
    def raw(self):
        return self._raw

    @property
    def labels(self):
        return self._labels

    @property
    def embed_dims(self):
        return self._wv.vector_size

    @property
    def doc_len(self):
        return self._doc_len

    def __getitem__(self, item):
        return self._raw[item], self._labels[item]

    def save(self, save_path):
        f = open(save_path, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_path):
        f = open(load_path, 'rb')
        return pickle.load(f, encoding='utf-8')
