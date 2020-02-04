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

from liv_ms.data import mona
from liv_ms.spectra import get_spectra, similarity
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
        filename = os.path.join(*[Path(__file__).parents[3],
                                  'data/'
                                  'MoNA-export-LC-MS-MS_Positive_Mode.json'])

        # Oxolinic acid, Flumequine false positive:
        df = mona.get_spectra(filename, 100).loc[[59, 51]]
        specs = get_spectra(df)

        scores = []

        for matcher_cls in _get_matchers():
            spec = specs[0].copy()
            query = specs[1].copy()

            matcher = matcher_cls([spec])
            result = matcher.search([query])

            scores.append(result[0][0])

        np.testing.assert_allclose(
            scores,
            [0.003206, 0.002632, 0.148492, 0.134185, 0.995261, 0.994394,
             0.243953, 0.606142, 0.134185],
            atol=0.01)


def _get_spectra(num_spectra, num_peaks):
    '''Get spectra, sorted by mass.'''
    spec = np.random.rand(num_spectra, num_peaks, 2)
    return np.sort(spec, axis=1)


def _get_matchers():
    '''Get matchers.'''
    return [partial(similarity.SimpleSpectraMatcher,
                    mass_acc=float('inf'), scorer=np.max),
            partial(similarity.SimpleSpectraMatcher,
                    mass_acc=float('inf'), scorer=np.average),
            partial(similarity.SimpleSpectraMatcher,
                    mass_acc=0.1, scorer=np.max),
            partial(similarity.SimpleSpectraMatcher,
                    mass_acc=0.1, scorer=np.average),
            partial(similarity.SimpleSpectraMatcher,
                    mass_acc=0.01, scorer=np.max),
            partial(similarity.SimpleSpectraMatcher,
                    mass_acc=0.01, scorer=np.average),
            similarity.BinnedSpectraMatcher,
            partial(similarity.KDTreeSpectraMatcher, use_i=True),
            partial(similarity.KDTreeSpectraMatcher, use_i=False)]


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
