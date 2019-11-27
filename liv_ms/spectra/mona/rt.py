'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
import ast
import re

from liv_ms.spectra import mona
import numpy as np
import pandas as pd


_FLOW_RATE_PATTERN = r'(\d+(?:\.\d+)?)\s?([um])[lL]\s?/\s?min' + \
    r'(?:\s+at\s+(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))? min)?'


def get_rt_data(filename, num_spec=float('inf'), regenerate_stats=True):
    '''Get RT data.'''
    if regenerate_stats:
        # Get spectra:
        df = mona.get_spectra(filename, num_spec=num_spec)

        # Clean data
        df = _clean_ms_level(df)
        df = _clean_rt(df)
        df = _clean_flow_rate(df)

        # Get stats:
        stats_df = _get_stats(df)

        # Save stats_df:
        _save_stats(stats_df)
    else:
        stats_df = pd.read_csv(
            'rt_stats.csv',
            converters={'flow rate values': ast.literal_eval})

    return stats_df


def _save_stats(stats_df):
    '''Save stats.'''
    # Convert flow rate values to list to enable saving:
    stats_df.loc[:, 'flow rate values'] = \
        stats_df['flow rate values'].apply(
        lambda x: x if isinstance(x, float) else list(x))

    stats_df.to_csv('rt_stats.csv', index=False)


def _clean_ms_level(df):
    '''Clean MS level.'''
    return df[df['ms level'] == 'MS2']


def _clean_rt(df):
    '''Clean retention time.'''
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


def _clean_flow_rate(df):
    '''Clean flow rate.'''
    df.loc[:, 'flow rate values'] = \
        df['flow rate'].apply(_clean_flow_rate_row)

    return df


def _clean_flow_rate_row(val):
    '''Clean single flow rate value.'''
    terms = []

    try:
        terms.extend([(0.0, float(val)),
                      (2**16, float(val))])
    except ValueError:

        val = val.lower()

        for term in val.split(','):
            term = term.strip()
            term = term.replace('min-1', '/min')
            mtch = re.match(_FLOW_RATE_PATTERN, term)

            if mtch:
                grps = mtch.groups()
                factor = 1.0 if grps[1] == 'm' else 1000.0
                rate = float(grps[0]) / factor

                if grps[2]:
                    terms.extend([(float(grps[2]), rate)])
                else:
                    terms.extend([(0.0, rate),
                                  (2**16, rate)])

                if grps[3]:
                    terms.extend([(float(grps[3]), rate)])
            else:
                terms.extend([(0.0, float('NaN')),
                              (2**16, float('NaN'))])

    return _get_flow_rate_grad(list(zip(*terms)))


def _get_flow_rate_grad(terms, steps=range(60)):
    '''Get flow rate gradient.'''
    times, rates = terms

    # Special case: if first rate is NaN:
    if np.isnan(rates[0]):
        return float('NaN')

    flow_rate_vals = []

    terms = zip(*terms)

    for step in steps:
        idx = np.searchsorted(times, step)

        if idx == 0:
            val = rates[0]
        elif idx >= len(rates):
            val = rates[-1]
        else:
            coeff = np.polyfit(times[idx - 1:idx + 1],
                               rates[idx - 1:idx + 1], 1)
            polynomial = np.poly1d(coeff)
            val = polynomial(step)

        flow_rate_vals.append(val)

    return flow_rate_vals


def _get_stats(df):
    '''Get retention time statistics.'''

    # Convert to tuples to enable hashing / grouping:
    df.loc[:, 'flow rate values'] = df['flow rate values'].apply(
        lambda x: x if isinstance(x, float) else tuple(x))

    stats_df = df.groupby(['name', 'smiles', 'column',
                           'flow rate values']).agg(
        {'retention time': ['mean', 'std']})

    # Flatten multi-index columns:
    stats_df.columns = [' '.join(col)
                        for col in stats_df.columns.values]

    # Reset multi-index index:
    return stats_df.reset_index()
