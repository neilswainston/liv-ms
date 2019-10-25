'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=wrong-import-order
import unittest
from liv_ms import similarity
import numpy as np


class TestSpectraMatcher(unittest.TestCase):
    '''Class to test methods in SpectraMatcher class.'''

    def test_search(self):
        '''Test search method of SpectraMatcher class.'''
        num_spectra = 2
        num_queries = 8
        num_spec_peaks = 6
        num_query_peaks = 48

        # Get spectra:
        spectra = _get_spectra(num_spectra, num_spec_peaks)

        matcher = similarity.SpectraMatcher(spectra)

        queries = _get_spectra(num_queries, num_query_peaks)

        result = matcher.search(queries)

        self.assertEqual(result.shape, (num_queries, num_spectra))

    def test_search_specific(self):
        '''Test search method of SpectraMatcher class.'''
        spectra = np.array([[[1, 0.2], [10, 0.3]],
                            [[1, 0.2], [10, 0.3]],
                            [[1, 0.2], [10, 0.3]]])

        matcher = similarity.SpectraMatcher(spectra)

        queries = np.array([[[1, 0.2], [10, 0.3]]])
        result = matcher.search(queries)

        self.assertAlmostEqual(result[0][0], 0, 12)


def _get_spectra(num_spectra, num_peaks):
    '''Get spectra.'''
    return np.random.rand(num_spectra, num_peaks, 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
