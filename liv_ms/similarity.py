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
        self.__spec_trees = _get_spec_trees(spectra)

    def search(self, queries):
        '''Search.'''
        res = np.array([_get_similarity_scores(spec_tree, queries)
                        for spec_tree in self.__spec_trees]).T

        return res


def _get_spec_trees(spectra):
    '''Get KDTree for each spectrum in spectra.'''
    return [KDTree(spec) for spec in spectra]


def _get_similarity_scores(spec_tree, queries):
    '''Get similarity score.'''
    return np.mean(spec_tree.query(queries)[0], axis=1)
