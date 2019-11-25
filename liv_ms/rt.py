'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
import sys

from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from rdkit import Chem
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from liv_ms.chem import encode
from liv_ms.plot import plot_loss, plot_scatter
from liv_ms.spectra import mona
import numpy as np
import pandas as pd

# from keras.constraints import maxnorm


def parse(filename, num_spec=float('inf')):
    '''Parse.'''
    # Get spectra:
    df = mona.get_spectra(filename, num_spec=num_spec)

    # Clean data
    df = _clean_ms_level(df)
    df = _clean_rt(df)

    # Get stats:
    stats_df = _get_stats(df)

    return df, stats_df


def _clean_ms_level(df):
    '''Clean MS level.'''
    return df[df['ms level'] == 'MS2']


def _clean_rt(df):
    '''Get spectra.'''
    df = df.dropna(subset=['retention time'])

    res = df['retention time'].apply(_clean_rt_row)
    df.loc[:, 'retention time'] = res
    df.loc[:, 'retention time'] = df['retention time'].astype('float32')

    return df.dropna(subset=['retention time'])


def _clean_rt_row(val):
    '''Clean single retention time value.'''
    try:
        val = val.replace('N/A', 'NaN')
        val = val.replace('min', '')

        if 's' in val:
            val = val.replace('sec', '')
            val = val.replace('s', '')
            return float(val) / 60.0
    except AttributeError:
        # Forgiveness, not permission. Assume float and pass:
        pass

    try:
        return float(val)
    except ValueError:
        return float('NaN')


def _get_stats(df):
    '''Get retention time statistics.'''
    stats_df = df.groupby(['name', 'smiles', 'column', 'flow rate']).agg(
        {'retention time': ['mean', 'std']})

    # Flatten multi-index columns:
    stats_df.columns = [' '.join(col)
                        for col in stats_df.columns.values]

    # Reset multi-index index:
    return stats_df.reset_index()


def _encode_x(df, fngrprnt_func):
    '''Encode features.'''
    encode_fnc = partial(encode, fngrprnt_func=fngrprnt_func)

    # Encode smiles;
    smiles = np.array([encode_fnc(s) for s in df['smiles']])

    # One-hot encode column
    _, column = _one_hot_encode(df['column'])

    return np.concatenate([smiles, column], axis=1)


def _one_hot_encode(values):
    '''One-hot encode values.'''
    label_encoder = LabelEncoder()
    return label_encoder, to_categorical(label_encoder.fit_transform(values))


def _create_train_model(X, y_scaled):
    '''Create and train model.'''
    # Create model:
    model = _create_model(X.shape[1])

    # Split data:
    X_train, X_dev, y_train, y_dev = train_test_split(X, y_scaled,
                                                      train_size=0.9)

    # Train model:
    return _train_model(model, X_train, X_dev, y_train, y_dev)


def _create_model(input_dim, loss='mean_squared_error', optimizer_func=Adam):
    '''Create model.'''
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(loss=loss,
                  optimizer=optimizer_func())

    return model


def _train_model(model, X_train, X_dev, y_train, y_dev):
    '''Train model.'''
    # Fit:
    history = model.fit(X_train, y_train,
                        validation_data=(X_dev, y_dev),
                        epochs=256)

    # Evaluate:
    train_mse = model.evaluate(X_train, y_train)
    test_mse = model.evaluate(X_dev, y_dev)

    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

    # Plot loss during training:
    plot_loss(history)

    return y_dev, model.predict(X_dev)


def _k_fold(X, y):
    '''k-fold.'''
    model_func = partial(_create_model, input_dim=X.shape[1])
    regressor = KerasRegressor(build_fn=model_func, epochs=256, batch_size=32,
                               verbose=1)

    estimators = []
    estimators.append(('scaler', MinMaxScaler()))

    estimators.append(('regression', regressor))

    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=16)
    results = cross_val_score(pipeline, X, y, cv=kfold)

    print('Train / test: %.3f (%.3f)' % (results.mean(), results.std()))

    # Plot predictions on validation data:
    plot_scatter(y.flatten(),
                 regressor.predict(X).flatten(),
                 'RT',
                 'RT measured / min',
                 'RT predicted / min')


def main(args):
    '''main method.'''
    # Get data:
    # _, stats_df = parse(args[0])
    # stats_df.to_csv('rt_stats.csv')
    stats_df = pd.read_csv('rt_stats.csv')

    # Filter data:
    stats_df = stats_df[stats_df['retention time mean'] < 12.0]

    # Encode data:
    X = _encode_x(stats_df, Chem.RDKFingerprint)

    # Scale data:
    y = stats_df['retention time mean'].to_numpy()
    y = y.reshape(len(y), 1)

    # _k_fold(X, y)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)
    y_dev, y_dev_pred = _create_train_model(X, y_scaled)

    # Plot predictions on validation data:
    plot_scatter(y_scaler.inverse_transform(y_dev).flatten(),
                 y_scaler.inverse_transform(y_dev_pred).flatten(),
                 'RT',
                 'RT measured / min',
                 'RT predicted / min')


if __name__ == '__main__':
    main(sys.argv[1:])
