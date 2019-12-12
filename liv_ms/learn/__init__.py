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
import numpy as np


def one_hot_encode(values):
    '''One-hot encode values.'''
    label_encoder = LabelEncoder()
    return label_encoder, to_categorical(label_encoder.fit_transform(values))


def k_fold(X, y, estimator, title, y_scaler, k, do_fit=True):
    '''Roll-your-own k-fold.'''
    # Perform k-fold fits:
    y_devs = []
    y_dev_preds = []
    train_mses = []
    test_mses = []

    for fold_idx in range(k):
        fold_title = '%s, Fold: %i' % (title, fold_idx + 1)

        X_train, X_dev, y_train, y_dev = \
            train_test_split(X, y, train_size=0.95)

        if do_fit:
            history = estimator.fit(X_train, y_train)
            train_mses.append(estimator.score(X_train, y_train))
            plot_loss(history, 'Loss: %s' % fold_title)

        test_mses.append(estimator.score(X_dev, y_dev))
        y_devs.extend(y_dev.flatten())
        y_dev_preds.extend(estimator.predict(X_dev).flatten())

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
