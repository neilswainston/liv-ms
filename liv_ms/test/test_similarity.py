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
        spectra = [[[random.random() * 100, random.random()]
                    for _ in range(16)]
                   for _ in range(256)]

        matcher = similarity.SpectraMatcher(spectra)

        queries = [[[random.random() * 100, random.random()]
                    for _ in range(16)]
                   for _ in range(32)]

        result = matcher.search(queries)

        self.assertEqual(result.shape, (32, 256))

    def test_search_specific(self):
        '''Test search method of similarity module.'''
        spectra = [[[1, 0.5], [10, 0.5]],
                   [[1, 0.5], [10, 0.5]],
                   [[1, 0.5], [10, 0.5]]]

        matcher = similarity.SpectraMatcher(spectra)

        queries = [[[1, 0.5], [10, 0.5]]]
        result = matcher.search(queries)

        self.assertAlmostEqual(result[0][0], 0, 12)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
