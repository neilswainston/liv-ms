'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
import sys

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler

from liv_ms.chem import encode_desc, encode_fngrprnt, get_fngrprnt_funcs
from liv_ms.learn import k_fold, nn, one_hot_encode
from liv_ms.spectra.mona.rt import get_rt_data
from liv_ms.utils import to_str
import numpy as np


def get_data(filename, regenerate_stats, scaler_func=MinMaxScaler):
    '''Get data.'''

    # Get data:
    stats_df = get_rt_data(filename,
                           regenerate_stats=regenerate_stats)

    X = np.concatenate(
        [_encode_chrom(stats_df), _encode_desc(stats_df)], axis=1)

    y = stats_df['retention time mean'].to_numpy()
    y = y.reshape(len(y), 1)

    y_scaler = scaler_func()
    y_scaled = y_scaler.fit_transform(y)
    y_scaled = y_scaled.ravel()

    return stats_df, X, y_scaled, y_scaler


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


def main(args):
    '''main method.'''
    # Get data:
    filename = args[0]
    regenerate_stats = bool(int(args[1]))
    verbose = int(args[2])
    k = 16

    stats_df, X, y, y_scaler = get_data(filename, regenerate_stats)

    for fngrprnt_func in get_fngrprnt_funcs():
        fngrprnt_enc = np.array([encode_fngrprnt(s, fngrprnt_func)
                                 for s in stats_df['smiles']])
        X = np.concatenate([X, fngrprnt_enc], axis=1)
        X = MinMaxScaler().fit_transform(X)
        title = to_str(fngrprnt_func)

        # Perform k-fold:
        for n_estimators in [10, 100]:
            estimator = RandomForestRegressor(n_estimators=n_estimators)
            # res = cross_val_score(estimator, X, y,
            #                      cv=KFold(n_splits=k))

            # print('%s: k-fold: Train / test: %.3f +/- %.3f' %
            #      (title, np.abs(res.mean()), res.std()))

            k_fold(X, y, estimator,
                   '%s_%s' % (type(estimator).__name__, title),
                   y_scaler, k=k)


if __name__ == '__main__':
    main(sys.argv[1:])
