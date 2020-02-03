'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import numpy as np


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
