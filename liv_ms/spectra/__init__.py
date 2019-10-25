'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
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
