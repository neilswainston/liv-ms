'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
import sys

from liv_ms.chem import encode, get_fngrprnt_funcs
from liv_ms.learn import fit, k_fold, one_hot_encode
from liv_ms.plot import plot_loss, plot_scatter
from liv_ms.spectra.mona.rt import get_rt_data
from liv_ms.utils import to_str
import numpy as np


def get_data(filename, regenerate_stats):
    '''Get data.'''

    # Get data:
    stats_df = get_rt_data(filename,
                           regenerate_stats=regenerate_stats)

    # Filter data:
    return stats_df[stats_df['retention time mean'] < 12.0]


def encode_data(stats_df, fngrprnt_func):
    '''Encode data.'''
    X = _encode_x(stats_df, fngrprnt_func)

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
    verbose = int(args[2])

    stats_df = get_data(filename, regenerate_stats)

    for fngrprnt_func in get_fngrprnt_funcs():
        X, y = encode_data(stats_df, fngrprnt_func)
        title = to_str(fngrprnt_func)

        # Perform k-fold:
        # res = k_fold(X, y)

        # print('k-fold: Train / test: %.3f (%.3f)' % (res.mean(), res.std()))

        # Perform k-fold fits:
        y_devs = []
        y_dev_preds = []

        for fold_idx in range(3):
            y_dev, y_dev_pred, history, train_mse, test_mse = \
                fit(X, y, epochs=256, verbose=verbose)

            y_devs.extend(y_dev.flatten())
            y_dev_preds.extend(y_dev_pred.flatten())

            # Plot loss during training:
            fold_title = '%s, Fold: %i' % (title, fold_idx + 1)
            plot_loss(history, 'Loss: %s' % fold_title)

            # print('%s: Train: %.3f, Test: %.3f' %
            #      (fold_title, train_mse, test_mse))

        # Plot predictions on validation data:
        label = plot_scatter(y_devs,
                             y_dev_preds,
                             'RT: %s' % title,
                             'RT measured / min',
                             'RT predicted / min')

        print('%s: Fit: %s' % (title, label))


if __name__ == '__main__':
    main(sys.argv[1:])
