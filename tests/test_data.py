import unittest
import os


class TestDatasets(unittest.TestCase):
    _DATASETS = ['data/amazon', 'data/movies']
    # _DATASETS = ['data/amazon']
    
    def test_exists(self):
        for set in self._DATASETS:
            self.assertTrue(os.path.exists(set + '/train.tsv'))
            self.assertTrue(os.path.exists(set + '/test.tsv'))
            self.assertTrue(os.path.exists(set + '/val.tsv'))

    def test_ratio(self):
        for ds in self._DATASETS:
            print('Testing %s...' % ds)
            train = self._load_data(ds + '/train.tsv')
            test = self._load_data(ds + '/test.tsv')
            val = self._load_data(ds + '/val.tsv')

            # Test set ratios
            total_len = len(train) + len(test) + len(val)
            delta = total_len * 0.00001
            self.assertAlmostEqual(total_len * 0.7, len(train), delta=delta)
            self.assertAlmostEqual(total_len * 0.15, len(val), delta=delta)
            self.assertAlmostEqual(total_len * 0.15, len(test), delta=delta)

            # Test label ratio
            total_pos = self._pos_ratio(train + test + val)
            train_pos = self._pos_ratio(train)
            self.assertAlmostEqual(total_pos, train_pos, delta=0.05)
            test_pos = self._pos_ratio(test)
            self.assertAlmostEqual(total_pos, test_pos, delta=0.05)
            val_pos = self._pos_ratio(val)
            self.assertAlmostEqual(total_pos, val_pos, delta=0.05)

    def _load_data(self, path):
        f = open(path, 'rt')
        data = []
        for i, line in enumerate(f):
            cols = line.split('\t')
            self.assertEqual(len(cols), 2, msg='Incorrect column number at line %d: %s' % (i + 1, cols))
            self.assertIn(cols[0], ['Positive', 'Negative'], msg='Unknown label at line %d' % (i + 1))
            data.append(cols)
        f.close()
        return data

    @staticmethod
    def _pos_ratio(dataset):
        total = len(dataset)
        pos = 0.0
        for point in dataset:
            if point[0] == 'Positive':
                pos += 1
        return pos / float(total)


if __name__ == '__main__':
    unittest.main()
