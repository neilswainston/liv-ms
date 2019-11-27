'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
import sys

from rdkit import Chem

from liv_ms.chem import encode
from liv_ms.learn import fit, k_fold, one_hot_encode
from liv_ms.plot import plot_scatter
from liv_ms.spectra.mona.rt import get_rt_data
import numpy as np


def get_data(filename, regenerate_stats):
    '''Get data.'''

    # Get data:
    stats_df = get_rt_data(filename,
                           regenerate_stats=regenerate_stats)

    # Filter data:
    stats_df = stats_df[stats_df['retention time mean'] < 12.0]

    # Encode data:
    X = _encode_x(stats_df, Chem.RDKFingerprint)

    # Scale data:
    y = stats_df['retention time mean'].to_numpy()
    y = y.reshape(len(y), 1)

    return X, y


def _encode_x(df, fngrprnt_func):
    '''Encode features.'''

    # Encode smiles;
    encode_fnc = partial(encode, fngrprnt_func=fngrprnt_func)
    smiles = np.array([encode_fnc(s) for s in df['smiles']])

    # One-hot encode column:
    _, column = one_hot_encode(df['column'])

    # Update flow rate:
    flow_rate_vals = np.array([np.array(vals)
                               for vals in df['flow rate values']])

    return np.concatenate([smiles, column, flow_rate_vals], axis=1)


def main(args):
    '''main method.'''
    # Get data:
    filename = args[0]
    regenerate_stats = bool(int(args[1]))

    X, y = get_data(filename, regenerate_stats)

    # Perform k-fold:
    results = k_fold(X, y)

    print('k-fold: Train / test: %.3f (%.3f)' %
          (results.mean(), results.std()))

    # Perform single fit:
    y_dev, y_dev_pred = fit(X, y)

    # Plot predictions on validation data:
    plot_scatter(y_dev.flatten(),
                 y_dev_pred.flatten(),
                 'RT',
                 'RT measured / min',
                 'RT predicted / min')


if __name__ == '__main__':
    main(sys.argv[1:])
