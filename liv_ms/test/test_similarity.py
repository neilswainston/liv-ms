'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=wrong-import-order
import unittest
from liv_ms import similarity
import numpy as np


class TestSimilarity(unittest.TestCase):
    '''Class to test methods in similarity module.'''

    def test_normalise_spectra_simple(self):
        '''Test normalise_spectra method of similarity module.'''
        spectra = np.array([[[1.0, 6.0], [3.0, 4.0]]])
        max_mass = similarity.normalise_spectra(spectra)

        # Test max_mass:
        self.assertEqual(max_mass, 3.0)

        # Test m/z:
        np.testing.assert_allclose(spectra[0][:, 0],
                                   [0.3333, 1.0],
                                   rtol=1e-3)

        # Test I:
        np.testing.assert_allclose(spectra[0][:, 1],
                                   [0.6, 0.4],
                                   rtol=1e-3)

    def test_normalise_spectra_complex(self):
        '''Test normalise_spectra method of similarity module.'''
        spectra = np.array([
            [[1.0, 6.0], [3.0, 4.0]],
            [[5.0, 0.02], [10.0, 0.08]]])

        max_mass = similarity.normalise_spectra(spectra)

        # Test max_mass:
        self.assertEqual(max_mass, 10.0)

        # Test m/z:
        np.testing.assert_allclose(spectra[0][:, 0],
                                   [0.1, 0.3],
                                   rtol=1e-3)

        np.testing.assert_allclose(spectra[1][:, 0],
                                   [0.5, 1.0],
                                   rtol=1e-3)

        # Test I:
        np.testing.assert_allclose(spectra[1][:, 1],
                                   [0.2, 0.8],
                                   rtol=1e-3)

    def test_normalise_spectra_max_mz(self):
        '''Test normalise_spectra method of similarity module.'''
        spectra = np.array([
            [[1.0, 6.0], [3.0, 4.0]],
            [[5.0, 0.02], [10.0, 0.08]]])

        max_mass = similarity.normalise_spectra(spectra, 100.0)

        # Test max_mass:
        self.assertEqual(max_mass, 100.0)

        # Test m/z:
        np.testing.assert_allclose(spectra[0][:, 0],
                                   [0.01, 0.03],
                                   rtol=1e-3)

        np.testing.assert_allclose(spectra[1][:, 0],
                                   [0.05, 0.1],
                                   rtol=1e-3)

        # Test I:
        np.testing.assert_allclose(spectra[1][:, 1],
                                   [0.2, 0.8],
                                   rtol=1e-3)


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
