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
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from rdkit import Chem
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from liv_ms.learn import nn
from liv_ms.plot import plot_loss, plot_scatter
from liv_ms.spectra.mona.rt import get_rt_data


def one_hot_encode(values):
    '''One-hot encode values.'''
    label_encoder = LabelEncoder()
    return label_encoder, to_categorical(label_encoder.fit_transform(values))


def fit(X, y, estimator, train_size=0.95, verbose=1):
    '''Fit data.'''

    # Split data:
    X_train, X_dev, y_train, y_dev = \
        train_test_split(X, y, train_size=train_size)

    # Fit:
    history = estimator.fit(X_train, y_train,
                            validation_data=(X_dev, y_dev),
                            verbose=verbose)

    # Evaluate:
    train_mse = estimator.score(X_train, y_train)
    test_mse = estimator.score(X_dev, y_dev)

    return y_dev, estimator.predict(X_dev), history, train_mse, test_mse
