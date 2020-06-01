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
from functools import partial

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


class SimpleSpectraMatcher(SpectraMatcher):
    '''Class to match spectra.'''

    def __init__(self, specs, mass_acc=0.01, scorer=np.max):
        super(SimpleSpectraMatcher, self).__init__(specs)
        self.__spectra, self.__max_mz = spectra.normalise(specs)
        self.__spectra = spectra.pad(self.__spectra)
        self.__mass_acc = mass_acc
        self.__scorer = scorer

    def search(self, queries):
        '''Search.'''
        queries, _ = spectra.normalise(queries, self.__max_mz)
        queries = spectra.pad(queries)

        # queries are masses (m/z), weights are intensities:
        fnc = partial(self.__get_sim_scores,
                      queries=self.__spectra[:, :, 0],
                      weights=self.__spectra[:, :, 1])

        lib_query_scores = np.apply_along_axis(fnc, 1, queries[:, :, 0])

        fnc = partial(self.__get_sim_scores,
                      queries=queries[:, :, 0],
                      weights=queries[:, :, 1])

        query_lib_scores = np.apply_along_axis(
            fnc, 1, self.__spectra[:, :, 0]).T

        return self.__scorer([query_lib_scores, lib_query_scores], axis=0)

    def __get_sim_scores(self, target, queries, weights):
        '''Get closest distances between query and spec,
        assuming sorted by m/z.'''
        # Only consider unpadded values:
        target = target[target > 0]

        len_target = len(target)
        sorted_idx = np.searchsorted(target, queries)
        sorted_idx[sorted_idx == len_target] = len_target - 1

        mask = (sorted_idx > 0) & \
            ((np.abs(queries - target[sorted_idx - 1])
              < np.abs(queries - target[sorted_idx])))

        dists = np.abs(queries - target[tuple([sorted_idx - mask])])
        dists[dists > self.__mass_acc / self.__max_mz] = 1

        return np.average(dists, weights=weights, axis=1)


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
        self.__spectra, self.__max_mz = spectra.normalise(specs)
        self.__use_i = use_i
        self.__mass_acc = mass_acc
        self.__inten_acc = inten_acc
        self.__spec_trees = self.__get_trees(self.__spectra)
        self.__spectra = spectra.pad(self.__spectra)

    def search(self, queries):
        '''Search.'''
        queries, _ = spectra.normalise(queries, self.__max_mz)
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


def main():
    '''main method.'''
    # Peaks have to be within this mass difference to be considered a match:
    mass_acc = 0.01

    # matcher matches query to target, and target to query, given two scores.
    # The 'scorer' function is applied to these two scores (in this case we
    # take the maximum (i.e. the worst of the two scores), but could be
    # replaced with a mean or weighted average:
    scorer = np.max

    # Target spectra, to be matched against:
    target_spec = np.array([
        [
            [89.87627, 5.0],
            [126.6, 12.9],
            [150.7, 100.0]
        ],
    ])

    # Query spectrum (may be a different length to the target_spec):
    query_spec = np.array([
        [
            [89.875, 12.6],
            [99.83, 67.9],
            [150.6987, 100.0]
        ],
    ])

    matcher = SimpleSpectraMatcher(target_spec, mass_acc, scorer)
    print(matcher.search(query_spec))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    main()
