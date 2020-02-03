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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler

from liv_ms.chem import encode_fngrprnt, get_fngrprnt_funcs
from liv_ms.data import rt, mona, metlin, shikifactory
from liv_ms.learn import k_fold  # , nn
from liv_ms.utils import to_str
import numpy as np


def main(args):
    '''main method.'''
    # Get data:
    filename = args[0]
    regen_stats = bool(int(args[1]))
    # verbose = int(args[2])
    k = 16

    stats_df, X, y, y_scaler = rt.get_data(
        filename, module=mona, regen_stats=regen_stats)

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
