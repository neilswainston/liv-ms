'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-few-public-methods
from collections import defaultdict

from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


class SpectraMatcher():
    '''Class to match spectra.'''

    def __init__(self, spectra, bin_size=0.1, min_val=0, max_val=1000):
        self.__bin_size = bin_size
        self.__min_val = min_val
        self.__max_val = max_val
        self.__num_bins = int((max_val - min_val) / self.__bin_size)

        self.__spec_matrix = self.__bin_spec(spectra)

    def search(self, query):
        '''Search.'''
        query_matrix = self.__bin_spec(query)
        return cosine_similarity(query_matrix, self.__spec_matrix)

    def __bin_spec(self, specs):
        '''Bin spectrum.'''
        row = []
        col = []
        data = []

        for spec_idx, spec in enumerate(specs):
            binned_spec = defaultdict(int)

            for mass, intensity in zip(*spec):
                if self.__min_val < mass < self.__max_val:
                    binned_mass = int(mass / self.__bin_size)
                    binned_spec[binned_mass] += intensity

            row.extend([spec_idx] * len(binned_spec))
            col.extend(binned_spec.keys())
            data.extend(binned_spec.values())

        return coo_matrix((data, (row, col)),
                          shape=(len(specs), self.__num_bins))
