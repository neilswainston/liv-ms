'''
Created on 15 Oct 2019

@author: neilswainston
'''
from collections import defaultdict
import random
import sys

from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


def main(args):
    '''main method.'''
    query = [[0.0, 1.2527, 9.765, 9.78], [0.25, 0.5, 0.75, 0.05]]
    query_matrix = bin_spec([query])

    specs = [[[random.random() * 10 for _ in range(16)],
              [random.random() for _ in range(16)]]
             for _ in range(256)] + [query]

    spec_matrix = bin_spec(specs)
    print(cosine_similarity(query_matrix, spec_matrix))


def bin_spec(specs, bin_size=0.1, min_val=0, max_val=10):
    '''Bin spectrum.'''
    num_bins = int((max_val - min_val) / bin_size)

    row = []
    col = []
    data = []

    for spec_idx, spec in enumerate(specs):
        binned_spec = defaultdict(int)

        for mass, intensity in zip(*spec):
            binned_mass = int(mass / bin_size)
            binned_spec[binned_mass] += intensity

        row.extend([spec_idx] * len(binned_spec))
        col.extend(binned_spec.keys())
        data.extend(binned_spec.values())

    return coo_matrix((data, (row, col)), shape=(len(specs), num_bins))


if __name__ == '__main__':
    main(sys.argv[1:])
