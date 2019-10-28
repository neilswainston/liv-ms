'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=arguments-differ
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
from abc import ABC, abstractmethod

from scipy.spatial import KDTree
from sklearn.metrics.pairwise import pairwise_distances

from liv_ms import spectra
import numpy as np


class SpectraMatcher(ABC):
    '''Class to match spectra.'''

    def __init__(self, spec, *kargs, **kwargs):
        pass

    @abstractmethod
    def search(self, queries, *kargs, **kwargs):
        '''Search.'''


class BinnedSpectraMatcher(SpectraMatcher):
    '''Class to match spectra.'''

    def __init__(self, specs, metric='euclidean',
                 bin_size=0.1, min_mz=0, max_mz=1000):
        super(BinnedSpectraMatcher, self).__init__(specs)
        self.__metric = metric
        self.__bin_size = bin_size
        self.__min_mz = min_mz
        self.__max_mz = max_mz
        self.__spec_matrix = spectra.bin_spec(specs,
                                              self.__bin_size,
                                              self.__min_mz,
                                              self.__max_mz)

        assert self.__spec_matrix.max()

    def search(self, queries):
        '''Search.'''
        query_matrix = spectra.bin_spec(queries,
                                        self.__bin_size,
                                        self.__min_mz,
                                        self.__max_mz)

        dists = pairwise_distances(query_matrix, self.__spec_matrix,
                                   metric=self.__metric)

        return dists / self.__spec_matrix.max()


class KDTreeSpectraMatcher(SpectraMatcher):
    '''Class to match spectra.'''

    def __init__(self, specs):
        super(KDTreeSpectraMatcher, self).__init__(specs)
        self.__max_mz = spectra.normalise(specs)
        self.__spectra = spectra.pad(specs)
        self.__spec_trees = [KDTree(s) for s in specs]

    def search(self, queries):
        '''Search.'''
        spectra.normalise(queries, self.__max_mz)
        query_trees = [KDTree(s) for s in queries]
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


# spec_trees = _get_spec_trees([[[1, 1]]])
# print(_get_sim_scores(spec_trees[0], np.array(
#    [[[0, 1e-16]]]), mass_accuracy=float('inf')))
