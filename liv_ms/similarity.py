'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-few-public-methods
from scipy.spatial import KDTree
import numpy as np


class SpectraMatcher():
    '''Class to match spectra.'''

    def __init__(self, spectra, min_val=0, max_val=1000):
        self.__min_val = min_val
        self.__max_val = max_val
        _normalise_intensities(spectra)
        self.__spec_trees = _get_spec_trees(spectra)

    def search(self, queries):
        '''Search.'''
        _normalise_intensities(queries)

        return np.array([_get_similarity_scores(spec_tree, queries)
                         for spec_tree in self.__spec_trees]).T


def _normalise_intensities(spectra):
    '''Normalise intensities.'''
    # Noamalise intensities:
    spectra[:, :, 1] = (spectra[:, :, 1].T /
                        spectra.sum(axis=1)[:, 1]).T


def _get_spec_trees(spectra):
    '''Get KDTree for each spectrum in spectra.'''
    return [KDTree(spec) for spec in spectra]


def _get_similarity_scores(spec_tree, queries):
    '''Get similarity score.'''
    dists = spec_tree.query(queries)[0]
    return np.average(dists, weights=queries[:, :, 1], axis=1)
