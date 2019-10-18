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
        num_spectra = 4
        num_queries = 8
        num_spec_peaks = 24
        num_query_peaks = 48

        spectra = [[[random.random() * 100, random.random()]
                    for _ in range(num_spec_peaks)]
                   for _ in range(num_spectra)]

        matcher = similarity.SpectraMatcher(spectra)

        queries = [[[random.random() * 100, random.random()]
                    for _ in range(num_query_peaks)]
                   for _ in range(num_queries)]

        result = matcher.search(queries)

        self.assertEqual(result.shape, (num_queries, num_spectra))

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
