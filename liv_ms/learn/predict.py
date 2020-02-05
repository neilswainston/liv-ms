'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
import sys

from rdkit.Chem.rdMolDescriptors import \
    GetHashedTopologicalTorsionFingerprintAsBitVect
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from liv_ms.chem import encode_fngrprnt
from liv_ms.data import rt
from liv_ms.data.rt import mona, shikifactory
from liv_ms.plot import plot_scatter
import numpy as np


def predict(df_train, X_train, y_train, df_test, X_test, y_scaler):
    '''Predict.'''
    X_train, x_scaler = _encode_x(df_train, X_train)
    X_test, _ = _encode_x(df_test, X_test, x_scaler=x_scaler)

    estimator = RandomForestRegressor()

    X_train, X_dev, y_train, y_dev = \
        train_test_split(X_train, y_train, train_size=0.95)

    estimator.fit(X_train, y_train)
    train_mse = estimator.score(X_train, y_train)

    dev_mse = _score(estimator, X_dev, y_dev, y_scaler, 'RT dev')
    print('Train: %.3f,  Dev: %.3f' % (train_mse, dev_mse))

    y_test_preds = _predict(estimator, X_test, y_scaler)
    print(y_test_preds)
    df_test['RT predicted'] = y_test_preds


def _score(estimator, X, y, y_scaler, label):
    '''Score.'''
    mse = estimator.score(X, y)

    y_inv = y_scaler.inverse_transform([[val] for val in y])
    y_preds_inv = _predict(estimator, X, y_scaler)

    # Plot predictions on validation data:
    fit = plot_scatter(y_inv,
                       y_preds_inv,
                       label,
                       'RT measured / min',
                       'RT predicted / min')

    print('%s fit: %s' % (label, fit))

    return mse


def _predict(estimator, X, y_scaler):
    '''Predict.'''
    y_preds = estimator.predict(X)

    return y_scaler.inverse_transform(
        [[val] for val in y_preds]).flatten()


def _encode_x(df, X, x_scaler=None):
    '''Encode x values.'''
    fngrprnt_func = GetHashedTopologicalTorsionFingerprintAsBitVect

    train_fngrprnt_enc = np.array([encode_fngrprnt(s, fngrprnt_func)
                                   for s in df['smiles']])
    X = np.concatenate([X, train_fngrprnt_enc], axis=1)

    if x_scaler:
        X = x_scaler.transform(X)
    else:
        x_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)

    return X, x_scaler


def main(args):
    '''main method.'''
    train_df, train_X, train_y, y_scaler = rt.get_data(
        args[0], module=mona, regen_stats=False)

    test_df, test_X, _, _ = rt.get_data(
        args[1], module=shikifactory, regen_stats=False,
        scaler_func=y_scaler)

    predict(train_df, train_X, train_y, test_df, test_X, y_scaler)

    test_df[['name', 'smiles', 'RT predicted']].to_csv('pred.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
