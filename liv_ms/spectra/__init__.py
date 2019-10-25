'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
from collections import defaultdict

from scipy.sparse import coo_matrix

import numpy as np


def normalise(spectra, max_mz=float('NaN')):
    '''Normalise spectra.'''
    if np.isnan(max_mz):
        max_mz = max([max(spec[:, 0]) for spec in spectra])

    for spec in spectra:
        # Normalise mz:
        spec[:, 0] = spec[:, 0] / max_mz

        # Normalise intensities:
        spec[:, 1] = spec[:, 1] / spec[:, 1].sum()

    return max_mz


def pad(spectra):
    '''Pad spectra.'''
    padded = []
    max_len = max([len(query) for query in spectra])

    for spec in spectra:
        padded.append(np.pad(spec,
                             [(0, max_len - len(spec)), (0, 0)],
                             'constant',
                             constant_values=0))

    return np.array(padded)


def bin_spec(specs, bin_size, min_val, max_val):
    '''Bin spectra.'''
    row = []
    col = []
    data = []

    num_bins = int((max_val - min_val) / bin_size)

    for spec_idx, spec in enumerate(specs):
        binned_spec = defaultdict(int)

        for mass, intensity in zip(*spec):
            if min_val < mass < max_val:
                binned_mass = int(mass / bin_size)
                binned_spec[binned_mass] += intensity

        row.extend([spec_idx] * len(binned_spec))
        col.extend(binned_spec.keys())
        data.extend(binned_spec.values())

    return coo_matrix((data, (row, col)),
                      shape=(len(specs), num_bins))
