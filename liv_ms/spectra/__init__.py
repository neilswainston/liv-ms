'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
from collections import defaultdict

from scipy.sparse import coo_matrix

import numpy as np


def get_spectra(df):
    '''Get spectra.'''
    return df.apply(_get_peaks, axis=1).to_numpy()


def _get_peaks(row):
    '''Get peaks.'''
    return np.column_stack(row[['m/z', 'I']])


def normalise(spectra, max_mz=float('NaN')):
    '''Normalise spectra.'''
    normalised = []

    if np.isnan(max_mz):
        max_mz = max([max(spec[:, 0]) for spec in spectra])

    for spec in spectra:
        # Clone:
        spec_copy = np.matrix.copy(spec)

        # Normalise mz:
        spec_copy[:, 0] = spec_copy[:, 0] / max_mz

        # Normalise intensities:
        spec_copy[:, 1] = spec_copy[:, 1] / spec_copy[:, 1].sum()

        # Reject masses > max_mass:
        normalised.append(spec_copy[spec_copy[:, 0] <= 1])

    return np.array(normalised), max_mz


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


def bin_spec(specs, bin_size, min_mz, max_mz):
    '''Bin spectra.'''
    row = []
    col = []
    data = []

    num_bins = int((max_mz - min_mz) / bin_size)

    for spec_idx, spec in enumerate(specs):
        binned_spec = defaultdict(int)

        for m_z, intensity in spec:
            if min_mz < m_z < max_mz:
                binned_mass = int((m_z - min_mz) / bin_size)
                binned_spec[binned_mass] += intensity

        row.extend([spec_idx] * len(binned_spec))
        col.extend(binned_spec.keys())
        data.extend(binned_spec.values())

    return coo_matrix((data, (row, col)),
                      shape=(len(specs), num_bins))
