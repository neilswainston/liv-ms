'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-many-arguments
from functools import partial

from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor


def get_regressor(input_dim,
                  output_dim,
                  hidden_layers=(128, 16),
                  loss='mean_squared_error',
                  optimizer_func=Adam,
                  dropout=0.2,
                  kernel_constraint=maxnorm(3),
                  bias_constraint=maxnorm(3),
                  batch_size=32,
                  epochs=512,
                  verbose=1):
    '''Get regressor.'''
    model_func = partial(_create_model,
                         input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_layers=hidden_layers,
                         loss=loss,
                         optimizer_func=optimizer_func,
                         dropout=dropout,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint)

    return KerasRegressor(build_fn=model_func,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose)


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
