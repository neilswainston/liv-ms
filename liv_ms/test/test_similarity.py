'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=wrong-import-order
from functools import partial
import os.path
from pathlib import Path
import unittest

from liv_ms import similarity, spectra
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
        for matcher_cls in _get_matchers():
            specs = _get_spectra(num_spectra, num_spec_peaks)
            queries = _get_spectra(num_queries, num_query_peaks)

            matcher = matcher_cls(specs)
            result = matcher.search(queries)

            self.assertEqual(result.shape, (num_queries, num_spectra))

    def test_search_equal(self):
        '''Test search method of SpectraMatcher class.'''
        for matcher_cls in _get_matchers():
            specs = np.array([[[1, 0.2], [10, 0.3]]])
            queries = np.copy(specs)

            matcher = matcher_cls(specs)
            result = matcher.search(queries)

            self.assertAlmostEqual(result[0][0], 0, 12)

    def test_search_distant(self):
        '''Test search method of SpectraMatcher class.'''
        for matcher_cls in _get_matchers():
            specs = np.array([[[800.0, 0.2]]])
            queries = np.array([[[1e-16, 1e-16], [8000.0, 0.2]]])

            matcher = matcher_cls(specs)
            result = matcher.search(queries)

            self.assertAlmostEqual(result[0][0], 1, 12)

    def test_search_real(self):
        '''Test search method of SpectraMatcher class.'''
        # Get spectra:
        filename = os.path.join(*[Path(__file__).parents[2],
                                  'data/'
                                  'MoNA-export-LC-MS-MS_Positive_Mode.json'])

        # Oxolinic acid, Flumequine false positive:
        df = spectra.mona.get_spectra(filename, 100).loc[[59, 51]]
        specs = spectra.get_spectra(df)

        scores = []

        for matcher_cls in _get_matchers():
            spec = specs[0].copy()
            query = specs[1].copy()

            matcher = matcher_cls([spec])
            result = matcher.search([query])

            scores.append(result[0][0])

        np.testing.assert_allclose(scores, [0.244, 0.604, 0.127], atol=0.01)


def _get_spectra(num_spectra, num_peaks):
    '''Get spectra.'''
    return np.random.rand(num_spectra, num_peaks, 2)


def _get_matchers():
    '''Get matchers.'''
    return [similarity.BinnedSpectraMatcher,
            partial(similarity.KDTreeSpectraMatcher, use_i=True),
            partial(similarity.KDTreeSpectraMatcher, use_i=False)]


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
