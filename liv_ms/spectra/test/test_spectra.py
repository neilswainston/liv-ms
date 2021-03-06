'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-self-use
# pylint: disable=wrong-import-order
import unittest

from liv_ms import spectra
import numpy as np


class TestSpectra(unittest.TestCase):
    '''Class to test methods in spectra module.'''

    def test_pad(self):
        '''Test pad method of spectra module.'''
        spec = np.array([
            [[1.0, 6.0]],
            [[5.0, 0.02], [10.0, 0.08]]])

        padded = spectra.pad(spec)

        self.assertEqual(len(padded[0]), 2)
        np.testing.assert_allclose(padded[0][1], [0.0, 0.0])

    def test_bin_spec(self):
        '''Test bin_spec method of spectra module.'''
        spec = np.array([
            [[1.17, 6.0], [1.18, 4.0], [1.9, 5.0], [2, 0.3]],
            [[0.9, 0.02], [10.0, 0.08]]])

        binned_spec = spectra.bin_spec(spec, 0.1, 1, 2)

        expected = [[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        np.testing.assert_allclose(binned_spec.toarray(),
                                   expected,
                                   rtol=1e-3)

    def test_normalise_simple(self):
        '''Test normalise method of spectra module.'''
        spec = np.array([[[1.0, 6.0], [3.0, 4.0]]])
        spec_normal, max_mass = spectra.normalise(spec)

        # Test max_mass:
        self.assertEqual(max_mass, 3.0)

        # Test m/z:
        np.testing.assert_allclose(spec_normal[0][:, 0],
                                   [0.3333, 1.0],
                                   rtol=1e-3)

        # Test I:
        np.testing.assert_allclose(spec_normal[0][:, 1],
                                   [0.6, 0.4],
                                   rtol=1e-3)

    def test_normalise_complex(self):
        '''Test normalise_ method of spectra module.'''
        spec = np.array([
            [[1.0, 6.0], [3.0, 4.0]],
            [[5.0, 0.02], [10.0, 0.08]]])

        spec_normal, max_mass = spectra.normalise(spec)

        # Test max_mass:
        self.assertEqual(max_mass, 10.0)

        # Test m/z:
        np.testing.assert_allclose(spec_normal[0][:, 0],
                                   [0.1, 0.3],
                                   rtol=1e-3)

        np.testing.assert_allclose(spec_normal[1][:, 0],
                                   [0.5, 1.0],
                                   rtol=1e-3)

        # Test I:
        np.testing.assert_allclose(spec_normal[1][:, 1],
                                   [0.2, 0.8],
                                   rtol=1e-3)

    def test_normalise_max_mz(self):
        '''Test normalise method of spectra module.'''
        spec = np.array([
            [[1.0, 6.0], [3.0, 4.0]],
            [[5.0, 0.02], [10.0, 0.08]]])

        spec_normal, max_mass = spectra.normalise(spec, 100.0)

        # Test max_mass:
        self.assertEqual(max_mass, 100.0)

        # Test m/z:
        np.testing.assert_allclose(spec_normal[0][:, 0],
                                   [0.01, 0.03],
                                   rtol=1e-3)

        np.testing.assert_allclose(spec_normal[1][:, 0],
                                   [0.05, 0.1],
                                   rtol=1e-3)

        # Test I:
        np.testing.assert_allclose(spec_normal[1][:, 1],
                                   [0.2, 0.8],
                                   rtol=1e-3)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
