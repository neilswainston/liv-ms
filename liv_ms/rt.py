'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
import sys

# from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from liv_ms.chem import encode
from liv_ms.plot import plot_loss
from liv_ms.spectra import mona
import numpy as np
import pandas as pd


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
    stats_df = df.groupby(['name', 'smiles']).agg(
        {'retention time': ['mean', 'std']})

    # Flatten multi-index columns:
    stats_df.columns = [' '.join(col)
                        for col in stats_df.columns.values]

    # Reset multi-index index:
    return stats_df.reset_index()


def _encode(df, fngrprnt_func):
    '''Encode chemicals.'''
    encode_fnc = partial(encode, fngrprnt_func=fngrprnt_func)
    df['X'] = df['smiles'].apply(encode_fnc)


def _create_model(input_dim):
    '''Create model.'''
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    return model


def _train_model(X, y):
    '''Train model.'''
    X_train, X_dev, y_train, y_dev = train_test_split(X, y,
                                                      train_size=0.9)

    model = _create_model(X_train.shape[1])

    model.compile(loss='mean_squared_error',
                  optimizer=Adam())

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


def main(args):
    '''main method.'''
    # Get data:
    # _, stats_df = parse(args[0])
    # stats_df.to_csv('rt_stats.csv')
    stats_df = pd.read_csv('rt_stats.csv')

    # Filter data:
    stats_df = stats_df[stats_df['retention time mean'] < 12.0]

    # Encode data:
    _encode(stats_df, Chem.RDKFingerprint)

    # Scale data:
    X = np.array(stats_df['X'].tolist())

    y = stats_df['retention time mean'].to_numpy()
    y = y.reshape(len(y), 1)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Train model:
    _train_model(X, y_scaled)


if __name__ == '__main__':
    main(sys.argv[1:])
