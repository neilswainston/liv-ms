'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
from functools import partial
import sys

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler

from liv_ms.chem import encode_desc, encode_fngrprnt, get_fngrprnt_funcs
from liv_ms.learn import fit, nn, one_hot_encode
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
    # return stats_df[stats_df['retention time mean'] < 12.0]
    return stats_df


def _encode_chrom(df):
    '''Encode chromatography.'''

    # One-hot encode column:
    _, column = one_hot_encode(df['column'])

    # Update flow rate:
    flow_rate_vals = np.array([np.array(vals)
                               for vals in df['flow rate values']])

    # Update gradient:
    gradient_vals = np.array([np.array(vals)
                              for vals in df['gradient values']])

    return np.concatenate([column, flow_rate_vals, gradient_vals], axis=1)


def _encode_desc(df):
    '''Encode descriptors.'''
    return np.array([encode_desc(s) for s in df['smiles']])


def _encode_fngrprnt(df, fngrprnt_func):
    '''Encode fingerprint.'''
    encode_fnc = partial(encode_fngrprnt, fngrprnt_func=fngrprnt_func)
    return np.array([encode_fnc(s) for s in df['smiles']])


def _k_fold(X, y, estimator, title, y_scaler, k, verbose):
    '''Roll-your-own k-fold.'''
    # Perform k-fold fits:
    y_devs = []
    y_dev_preds = []
    train_mses = []
    test_mses = []

    for fold_idx in range(k):
        y_dev, y_dev_pred, history, train_mse, test_mse = \
            fit(X, y, estimator, train_size=0.95, verbose=verbose)

        y_devs.extend(y_dev.flatten())
        y_dev_preds.extend(y_dev_pred.flatten())

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # Plot loss during training:
        fold_title = '%s, Fold: %i' % (title, fold_idx + 1)
        plot_loss(history, 'Loss: %s' % fold_title)

    y_devs_inv = y_scaler.inverse_transform([[val] for val in y_devs])
    y_dev_preds_inv = y_scaler.inverse_transform(
        [[val] for val in y_dev_preds])

    # Plot predictions on validation data:
    label = plot_scatter(y_devs_inv,
                         y_dev_preds_inv,
                         'RT: %s' % title,
                         'RT measured / min',
                         'RT predicted / min')

    print('%s: Fit: %s' % (title, label))
    print('%s: k-fold: Train: %.3f,  Test: %.3f' %
          (title, np.mean(train_mses), np.mean(test_mses)))


def main(args):
    '''main method.'''
    # Get data:
    filename = args[0]
    regenerate_stats = bool(int(args[1]))
    verbose = int(args[2])
    scaler_func = MinMaxScaler
    k = 16
    epochs = 32

    stats_df = get_data(filename, regenerate_stats)
    feat_enc = np.concatenate(
        [_encode_chrom(stats_df), _encode_desc(stats_df)], axis=1)

    y = stats_df['retention time mean'].to_numpy()
    y = y.reshape(len(y), 1)

    y_scaler = scaler_func()
    y_scaled = y_scaler.fit_transform(y)

    for fngrprnt_func in get_fngrprnt_funcs():
        fngrprnt_enc = _encode_fngrprnt(stats_df, fngrprnt_func)
        X = np.concatenate([feat_enc, fngrprnt_enc], axis=1)
        X = scaler_func().fit_transform(X)

        title = to_str(fngrprnt_func)

        # Perform k-fold:
        estimator = nn.get_regressor(X.shape[1],
                                     y_scaled.shape[1],
                                     kernel_constraint=None,
                                     bias_constraint=None,
                                     epochs=epochs,
                                     verbose=verbose)

        res = cross_val_score(estimator, X, y_scaled, cv=KFold(n_splits=16))

        print('%s: k-fold: Train / test: %.3f +/- %.3f' %
              (title, -res.mean(), res.std()))

        _k_fold(X, y_scaled, estimator, title, y_scaler, k, verbose)


if __name__ == '__main__':
    main(sys.argv[1:])
