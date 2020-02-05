'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
from sklearn.preprocessing import MinMaxScaler

from liv_ms.chem import encode_desc
import numpy as np


def get_data(filename, module, regen_stats, scaler_func=MinMaxScaler,
             max_rt=60.0, columns=None):
    '''Get data.'''
    # Get data:
    stats_df = module.get_rt_data(filename, regen_stats=regen_stats)

    if 'retention time mean' in stats_df.columns:
        stats_df = stats_df[stats_df['retention time mean'] < max_rt]

    X = np.concatenate(
        [_encode_chrom(stats_df, columns=columns),
         _encode_desc(stats_df)], axis=1)

    if 'retention time mean' in stats_df.columns:
        y = stats_df['retention time mean'].to_numpy()
        y = y.reshape(len(y), 1)

        if scaler_func:
            y_scaler = scaler_func()
            y_scaled = y_scaler.fit_transform(y)
        else:
            y_scaler = None
            y_scaled = y

        y_scaled = y_scaled.ravel()
    else:
        y_scaler, y_scaled = None, None

    return stats_df, X, y_scaled, y_scaler


def get_timecourse_vals(terms, steps=range(60)):
    '''Get timecourse values.'''
    times, vals = terms

    # Special case: if first val is NaN:
    if np.isnan(vals[0]):
        return float('NaN')

    timecourse_vals = []

    terms = zip(*terms)

    for step in steps:
        idx = np.searchsorted(times, step)

        if idx == 0:
            val = vals[0]
        elif idx >= len(vals):
            val = vals[-1]
        else:
            coeff = np.polyfit(times[idx - 1:idx + 1],
                               vals[idx - 1:idx + 1], 1)
            polynomial = np.poly1d(coeff)
            val = polynomial(step)

        timecourse_vals.append(val)

    return timecourse_vals


def get_stats(df):
    '''Get retention time statistics.'''

    # Convert to tuples to enable hashing / grouping:
    for col_name in ['column values',
                     'flow rate values',
                     'gradient values']:
        df.loc[:, col_name] = df[col_name].apply(
            lambda x: x if isinstance(x, float) else tuple(x))

    if 'retention time' in df.columns:
        stats_df = df.groupby(['smiles', 'column values',
                               'flow rate values', 'gradient values'])

        stats_df = stats_df.agg({'retention time': ['mean', 'std']})

        # Flatten multi-index columns:
        stats_df.columns = [' '.join(col)
                            for col in stats_df.columns.values]

        # Reset multi-index index:
        stats_df.reset_index(inplace=True)
    else:
        stats_df = df

    return stats_df


def _encode_chrom(df, columns=None):
    '''Encode chromatography.'''
    if columns is None:
        columns = ['column values',
                   'flow rate values',
                   'gradient values']

    arrays = []

    # One-hot encode column:
    if 'column values' in columns:
        arrays.append(np.array([np.array(vals)
                                for vals in df['column values']]))

    # Update flow rate:
    if 'flow rate values' in columns:
        arrays.append(np.array([np.array(vals)
                                for vals in df['flow rate values']]))

    # Update gradient:
    if 'gradient values' in columns:
        arrays.append(np.array([np.array(vals)
                                for vals in df['gradient values']]))

    return np.concatenate(arrays, axis=1) if arrays \
        else np.array([[] for _ in range(len(df))])


def _encode_desc(df):
    '''Encode descriptors.'''
    return np.array([encode_desc(s) for s in df['smiles']])
