import unittest

from sentiment.loader import SentimentDataLoader


class TestLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = SentimentDataLoader('data/movies')

    def test_load_data(self):
        self.loader.load_data()


if __name__ == '__main__':
    unittest.main()
