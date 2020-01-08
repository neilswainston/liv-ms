'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import ast
from liv_ms.data.mona import column
import pandas as pd


def get_rt_data(filename, module, num_spec=float('inf'), regen_stats=True):
    '''Get RT data.'''
    if regen_stats:
        # Get spectra:
        df = module.get_rt_data(filename, num_spec)

        # Encode column:
        column.encode_column(df)

        # Get stats:
        stats_df = _get_stats(df)

        # Save stats_df:
        _save_stats(stats_df)
    else:
        stats_df = pd.read_csv(
            'rt_stats.csv',
            converters={'column values': ast.literal_eval,
                        'flow rate values': ast.literal_eval,
                        'gradient values': ast.literal_eval})

    return stats_df


def _get_stats(df):
    '''Get retention time statistics.'''

    # Convert to tuples to enable hashing / grouping:
    for col_name in ['column values',
                     'flow rate values',
                     'gradient values']:
        df.loc[:, col_name] = df[col_name].apply(
            lambda x: x if isinstance(x, float) else tuple(x))

    df.to_csv('out.csv')

    stats_df = df.groupby(['name', 'smiles', 'column values',
                           'flow rate values', 'gradient values']).agg(
        {'retention time': ['mean', 'std']})

    # Flatten multi-index columns:
    stats_df.columns = [' '.join(col)
                        for col in stats_df.columns.values]

    # Reset multi-index index:
    return stats_df.reset_index()


def _save_stats(stats_df):
    '''Save stats.'''
    # Convert values to list to enable saving:
    for col_name in ['column values',
                     'flow rate values',
                     'gradient values']:
        stats_df.loc[:, col_name] = \
            stats_df[col_name].apply(
            lambda x: x if isinstance(x, float) else list(x))

    stats_df.to_csv('rt_stats.csv', index=False)
