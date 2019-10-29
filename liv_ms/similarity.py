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

    def __init__(self, specs, use_i, mass_acc=0.1, inten_acc=0.1):
        super(KDTreeSpectraMatcher, self).__init__(specs)
        self.__max_mz = spectra.normalise(specs)
        self.__spectra = spectra.pad(specs)
        self.__use_i = use_i
        self.__mass_acc = mass_acc
        self.__inten_acc = inten_acc
        self.__spec_trees = self.__get_trees(specs)

    def search(self, queries):
        '''Search.'''
        spectra.normalise(queries, self.__max_mz)
        query_trees = self.__get_trees(queries)
        queries = spectra.pad(queries)

        query_lib_scores = np.array(
            [self.__get_sim_scores(spec_tree,
                                   self.__get_data(queries),
                                   weights=queries[:, :, 1])
             for spec_tree in self.__spec_trees]).T

        lib_query_scores = np.array(
            [self.__get_sim_scores(query_tree,
                                   self.__get_data(self.__spectra),
                                   weights=self.__spectra[:, :, 1])
             for query_tree in query_trees])

        return (query_lib_scores + lib_query_scores) / 2

    def __get_trees(self, spec):
        '''Get KDTrees.'''
        # Return 2D trees: m/z and I:
        return [KDTree(s) for s in self.__get_data(spec)]

    def __get_data(self, specs):
        '''Get data.'''
        # Return 2D data: m/z and I:
        if self.__use_i:
            return specs

        # Return 1D data: m/z only:
        return np.array([spec[:, 0].reshape(spec.shape[0], 1)
                         for spec in specs])

    def __get_sim_scores(self, lib_spec_tree, queries, weights):
        '''Get similarity score.'''
        max_dist = np.sqrt(2) if self.__use_i else 1

        dist_upp_bnd = np.sqrt(
            (self.__mass_acc / self.__max_mz)**2 + self.__inten_acc**2) \
            if self.__use_i \
            else self.__mass_acc / self.__max_mz

        dists, _ = lib_spec_tree.query(
            queries,
            distance_upper_bound=dist_upp_bnd)

        dists[dists == np.inf] = max_dist

        return np.average(dists / max_dist, weights=weights, axis=1)
