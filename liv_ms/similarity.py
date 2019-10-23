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

    def __init__(self, spectra):
        self.__max_mz = _normalise_spectra(spectra)
        self.__spectra = _pad(spectra)
        self.__spec_trees = _get_spec_trees(spectra)

    def search(self, queries):
        '''Search.'''
        _normalise_spectra(queries, self.__max_mz)
        query_trees = _get_spec_trees(queries)
        queries = _pad(queries)

        query_lib_scores = np.array(
            [self.__get_sim_scores(spec_tree, queries)
             for spec_tree in self.__spec_trees]).T

        lib_query_scores = np.array(
            [self.__get_sim_scores(spec_tree, self.__spectra)
             for spec_tree in query_trees])

        return (query_lib_scores + lib_query_scores) / 2

    def __get_sim_scores(self, lib_spec_tree, queries,
                         mass_acc=0.1, inten_acc=0.1):
        '''Get similarity score.'''
        dists = lib_spec_tree.query(
            queries,
            distance_upper_bound=np.sqrt(
                mass_acc / self.__max_mz + inten_acc))[0]
        dists[dists == np.inf] = np.sqrt(2)
        return np.average(dists / np.sqrt(2), weights=queries[:, :, 1], axis=1)


def _normalise_spectra(spectra, max_mz=float('NaN')):
    '''Normalise spectra.'''
    if np.isnan(max_mz):
        max_mz = max([max(spec[:, 0]) for spec in spectra])

    for spec in spectra:
        # Normalise mz:
        spec[:, 0] = spec[:, 0] / max_mz

        # Normalise intensities:
        spec[:, 1] = spec[:, 1] / spec[:, 1].sum()

    return max_mz


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


# spec_trees = _get_spec_trees([[[1, 1]]])
# print(_get_sim_scores(spec_trees[0], np.array(
#    [[[0, 1e-16]]]), mass_accuracy=float('inf')))
