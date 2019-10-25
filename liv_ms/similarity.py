'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-few-public-methods
# pylint: disable=wrong-import-order
from scipy.spatial import KDTree

from liv_ms import spectra
import numpy as np


class SpectraMatcher():
    '''Class to match spectra.'''

    def __init__(self, spec):
        self.__max_mz = spectra.normalise(spec)
        self.__spectra = spectra.pad(spec)
        self.__spec_trees = _get_spec_trees(spec)

    def search(self, queries):
        '''Search.'''
        spectra.normalise(queries, self.__max_mz)
        query_trees = _get_spec_trees(queries)
        queries = spectra.pad(queries)

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


def _get_spec_trees(spec):
    '''Get KDTree for each spectrum in spectra.'''
    return [KDTree(s) for s in spec]


# spec_trees = _get_spec_trees([[[1, 1]]])
# print(_get_sim_scores(spec_trees[0], np.array(
#    [[[0, 1e-16]]]), mass_accuracy=float('inf')))
