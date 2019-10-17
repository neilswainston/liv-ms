'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
import random
import unittest

from liv_ms import similarity


class TestSimilarity(unittest.TestCase):
    '''Class to test methods in similarity module.'''

    def test_search(self):
        '''Test search method of similarity module.'''
        spectra = [[[random.random() * 10 for _ in range(16)],
                    [random.random() for _ in range(16)]]
                   for _ in range(256)]

        matcher = similarity.SpectraMatcher(spectra)

        query = [[0.0, 1.2527, 9.765, 9.78], [0.25, 0.5, 0.75, 0.05]]
        result = matcher.search([query])

        self.assertEqual(result.shape, (1, 256))

    def test_search_specific(self):
        '''Test search method of similarity module.'''
        spectra = [[[1, 2, 5], [1, 1, 1]]]

        matcher = similarity.SpectraMatcher(spectra)

        query = [[1, 20, 3], [1e-6, 1e-6, 1e-6]]
        result = matcher.search([query])

        self.assertAlmostEqual(result[0][0], 1 / 3, 12)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()