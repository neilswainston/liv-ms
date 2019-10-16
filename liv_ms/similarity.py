'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-few-public-methods
from collections import defaultdict
import random
import sys

from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


class SpectraMatcher():
    '''Class to match spectra.'''

    def __init__(self, bin_size=0.1, min_val=0, max_val=10):
        self.__bin_size = bin_size
        self.__num_bins = int((max_val - min_val) / self.__bin_size)

        spectra = [[[random.random() * 10 for _ in range(16)],
                    [random.random() for _ in range(16)]]
                   for _ in range(256)]

        self.__spec_matrix = self.__bin_spec(spectra)

    def search(self, query):
        '''Search.'''
        query_matrix = self.__bin_spec([query])
        return cosine_similarity(query_matrix, self.__spec_matrix)

    def __bin_spec(self, specs):
        '''Bin spectrum.'''
        row = []
        col = []
        data = []

        for spec_idx, spec in enumerate(specs):
            binned_spec = defaultdict(int)

            for mass, intensity in zip(*spec):
                binned_mass = int(mass / self.__bin_size)
                binned_spec[binned_mass] += intensity

            row.extend([spec_idx] * len(binned_spec))
            col.extend(binned_spec.keys())
            data.extend(binned_spec.values())

        return coo_matrix((data, (row, col)),
                          shape=(len(specs), self.__num_bins))


def main(args):
    '''main method.'''
    matcher = SpectraMatcher()

    query = [[0.0, 1.2527, 9.765, 9.78], [0.25, 0.5, 0.75, 0.05]]
    result = matcher.search(query)
    print(result)


if __name__ == '__main__':
    main(sys.argv[1:])
