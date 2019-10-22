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
        self.__spectra = _pad(spectra)
        self.__spec_trees = _get_spec_trees(spectra)

    def search(self, queries):
        '''Search.'''
        _normalise_intensities(queries)
        query_trees = _get_spec_trees(queries)
        queries = _pad(queries)

        query_lib_scores = np.array([_get_sim_scores(spec_tree, queries)
                                     for spec_tree in self.__spec_trees]).T

        lib_query_scores = np.array([_get_sim_scores(spec_tree, self.__spectra)
                                     for spec_tree in query_trees])

        return (query_lib_scores + lib_query_scores) / 2


def _normalise_intensities(spectra):
    '''Normalise intensities.'''
    # Noamalise intensities:
    for spec in spectra:
        spec[:, 1] = spec[:, 1] / spec[:, 1].sum()


def _pad(spectra):
    '''Pad spectra.'''
    padded = []
    max_len = max([len(query) for query in spectra])

    for spec in spectra:
        padded.append(np.pad(spec,
                             [(0, max_len - len(spec)), (0, 0)],
                             'constant',
                             constant_values=0))

    return np.array(padded)


def _get_spec_trees(spectra):
    '''Get KDTree for each spectrum in spectra.'''
    return [KDTree(spec) for spec in spectra]


def _get_sim_scores(lib_spec_tree, queries):
    '''Get similarity score.'''
    dists = lib_spec_tree.query(queries)[0]
    return np.average(dists, weights=queries[:, :, 1], axis=1)
