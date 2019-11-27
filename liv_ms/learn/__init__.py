'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
from functools import partial
import sys

from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from rdkit import Chem
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from liv_ms.chem import encode
from liv_ms.plot import plot_loss, plot_scatter
from liv_ms.spectra.mona.rt import get_rt_data


def one_hot_encode(values):
    '''One-hot encode values.'''
    label_encoder = LabelEncoder()
    return label_encoder, to_categorical(label_encoder.fit_transform(values))


def k_fold(X, y,
           n_splits=16,
           hidden_layers=(128, 16),
           loss='mean_squared_error',
           optimizer_func=Adam,
           batch_size=32,
           dropout=0.2,
           kernel_constraint=maxnorm(3),
           bias_constraint=maxnorm(3),
           epochs=512):
    '''k-fold.'''
    model_func = partial(create_model,
                         input_dim=X.shape[1],
                         output_dim=y.shape[1],
                         hidden_layers=hidden_layers,
                         loss=loss,
                         optimizer_func=optimizer_func,
                         dropout=dropout,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint)

    regressor = KerasRegressor(build_fn=model_func,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=1)

    estimators = []
    estimators.append(('scaler', MinMaxScaler()))
    estimators.append(('regression', regressor))

    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=n_splits)
    return cross_val_score(pipeline, X, y, cv=kfold)


def fit(X, y,
        train_size=0.95,
        hidden_layers=(128, 16),
        loss='mean_squared_error',
        optimizer_func=Adam,
        batch_size=32,
        dropout=0.2,
        kernel_constraint=maxnorm(3),
        bias_constraint=maxnorm(3),
        epochs=512,
        verbose=1):
    '''Fit data.'''
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    y_dev, y_dev_pred, history, train_mse, test_mse = \
        create_train_model(X, y_scaled,
                           train_size=train_size,
                           hidden_layers=hidden_layers,
                           loss=loss,
                           optimizer_func=optimizer_func,
                           batch_size=batch_size,
                           dropout=dropout,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint,
                           epochs=epochs,
                           verbose=verbose)

    return y_scaler.inverse_transform(y_dev), \
        y_scaler.inverse_transform(y_dev_pred), \
        history, train_mse, test_mse


def create_train_model(X, y, train_size,
                       hidden_layers, loss, optimizer_func, batch_size,
                       dropout, kernel_constraint, bias_constraint,
                       epochs,
                       verbose):
    '''Create and train model.'''
    # Create model:
    model = create_model(X.shape[1],
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
    return train_model(model, X_train, X_dev, y_train, y_dev,
                       batch_size=batch_size, epochs=epochs,
                       verbose=verbose)


def create_model(input_dim, output_dim, hidden_layers, loss, optimizer_func,
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


def train_model(model, X_train, X_dev, y_train, y_dev,
                batch_size, epochs,
                verbose):
    '''Train model.'''
    # Fit:
    history = model.fit(X_train, y_train,
                        validation_data=(X_dev, y_dev),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose)

    # Evaluate:
    train_mse = model.evaluate(X_train, y_train)
    test_mse = model.evaluate(X_dev, y_dev)

    return y_dev, model.predict(X_dev), history, train_mse, test_mse
