import unittest
import numpy as np


from sentiment.loader import SentimentDataLoader
from sentiment.model import SentimentNet


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = SentimentDataLoader('./data/movies/', cutoff=100, doc_len=300)
        cls.loader.load_data()
        cls.model = SentimentNet(cls.loader)

    def test_train(self):
        self.model.compile()
        self.model.train(epochs=1)

    def test_predict(self):
        r = self.model.predict('Hello World.')
        for p in r:
            self.assertIsInstance(p, np.float32)
        self.assertAlmostEqual(sum(r), 1.0, delta=0.001)

    def test_evaluate(self):
        r = self.model.evaluate()


if __name__ == '__main__':
    unittest.main()
