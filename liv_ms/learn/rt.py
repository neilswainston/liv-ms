'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
from functools import partial
import sys

from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from rdkit import Chem
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from liv_ms.chem import encode
from liv_ms.learn import one_hot_encode
from liv_ms.plot import plot_loss, plot_scatter
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


def fit(X, y,
        train_size=0.95,
        hidden_layers=(128, 16),
        loss='mean_squared_error',
        optimizer_func=Adam,
        dropout=0.2,
        kernel_constraint=maxnorm(3),
        bias_constraint=maxnorm(3),
        epochs=16):
    '''Fit data.'''
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    y_dev, y_dev_pred = \
        _create_train_model(X, y_scaled,
                            train_size=train_size,
                            hidden_layers=hidden_layers,
                            loss=loss,
                            optimizer_func=optimizer_func,
                            dropout=dropout,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint,
                            epochs=epochs)

    return y_scaler.inverse_transform(y_dev), \
        y_scaler.inverse_transform(y_dev_pred)


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


def _create_train_model(X, y, train_size,
                        hidden_layers, loss, optimizer_func,
                        dropout, kernel_constraint, bias_constraint,
                        epochs):
    '''Create and train model.'''
    # Create model:
    model = _create_model(X.shape[1],
                          y.shape[1],
                          hidden_layers=hidden_layers,
                          loss=loss,
                          optimizer_func=optimizer_func,
                          dropout=dropout,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)

    # Split data:
    X_train, X_dev, y_train, y_dev = \
        train_test_split(X, y, train_size=train_size)

    # Train model:
    return _train_model(model, X_train, X_dev, y_train, y_dev, epochs=epochs)


def _create_model(input_dim, output_dim, hidden_layers, loss, optimizer_func,
                  dropout, kernel_constraint, bias_constraint):
    '''Create model.'''
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(input_dim,)))

    for hidden_layer in hidden_layers:
        model.add(Dense(hidden_layer, activation='relu',
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint))
        model.add(Dropout(dropout))

    model.add(Dense(output_dim, activation='linear'))

    model.compile(loss=loss, optimizer=optimizer_func())

    return model


def _train_model(model, X_train, X_dev, y_train, y_dev, epochs):
    '''Train model.'''
    # Fit:
    history = model.fit(X_train, y_train,
                        validation_data=(X_dev, y_dev),
                        epochs=epochs)

    # Evaluate:
    train_mse = model.evaluate(X_train, y_train)
    test_mse = model.evaluate(X_dev, y_dev)

    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

    # Plot loss during training:
    plot_loss(history)

    return y_dev, model.predict(X_dev)


def _k_fold(X, y, batch_size=32, epochs=512, n_splits=16):
    '''k-fold.'''
    model_func = partial(_create_model, input_dim=X.shape[1])
    regressor = KerasRegressor(build_fn=model_func,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=1)

    estimators = []
    estimators.append(('scaler', MinMaxScaler()))
    estimators.append(('regression', regressor))

    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=n_splits)
    results = cross_val_score(pipeline, X, y, cv=kfold)

    print('k-fold: Train / test: %.3f (%.3f)' %
          (results.mean(), results.std()))

    # Plot predictions on validation data:
    plot_scatter(y.flatten(),
                 regressor.predict(X).flatten(),
                 'RT',
                 'RT measured / min',
                 'RT predicted / min')


def main(args):
    '''main method.'''
    # Get data:
    filename = args[0]
    regenerate_stats = bool(int(args[1]))

    X, y = get_data(filename, regenerate_stats)

    # _k_fold(X, y)

    y_dev, y_dev_pred = fit(X, y)

    # Plot predictions on validation data:
    plot_scatter(y_dev.flatten(),
                 y_dev_pred.flatten(),
                 'RT',
                 'RT measured / min',
                 'RT predicted / min')


if __name__ == '__main__':
    main(sys.argv[1:])
