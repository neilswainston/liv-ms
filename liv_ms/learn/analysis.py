'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
import json
import sys

from rdkit import Chem

import numpy as np
import pandas as pd


def get_bonds_frequency(df):
    '''Get bonds frequency.'''
    bonds_frequency = pd.Series(
        [item for sublist in df['bonds_broken']
         for item in sublist]).value_counts(normalize=True, dropna=False)

    bonds_frequency_df = bonds_frequency.reset_index()
    bonds_frequency_df['index'] = bonds_frequency_df['index'].apply(_decode)
    bonds_frequency_df.set_index('index', inplace=True)
    bonds_frequency_df.index.name = 'bonds'
    bonds_frequency_df.columns = ['frequency']

    return bonds_frequency_df


def get_data(filename):
    '''Get data.'''
    df = pd.read_csv(filename)
    df['m/z'] = df['m/z'].apply(_to_numpy, sep=',')
    df['METFRAG_MZ'] = df['METFRAG_MZ'].apply(_to_numpy, sep=',')
    df['METFRAG_BROKEN_BONDS'] = df['METFRAG_BROKEN_BONDS'].apply(_to_numpy_2d)

    df['match_idxs'] = df.apply(_match, axis=1)
    df['bonds_broken'] = df.apply(_get_broken_bonds, axis=1)

    return df


def _to_numpy(array_str, sep=','):
    '''Convert array_str to numpy.'''
    return np.fromstring(array_str[1:-1], sep=sep)


def _to_numpy_2d(array_str):
    '''Convert array_str to numpy 2d array.'''
    values = json.loads(array_str)
    return np.array([tuple(val) for val in values])


def _match(row, tol=0.01):
    '''Determine if masses match.'''
    match_idxs = np.ones(row['m/z'].size, dtype=int) * -1
    abs_diff = np.abs(np.subtract.outer(row['m/z'], row['METFRAG_MZ'])) < tol
    abs_diff_match_idxs = np.where(np.any(abs_diff, axis=1))
    abs_diff_idxs = np.argmax(abs_diff, axis=1)[abs_diff_match_idxs]
    np.put(match_idxs, abs_diff_match_idxs, abs_diff_idxs, mode='raise')
    return match_idxs


def _get_broken_bonds(row):
    '''Get bonds broken.'''
    broken_bonds = np.empty(row['match_idxs'].size, dtype=object)
    match_idxs = np.where(row['match_idxs'] > -1)
    bonds_idxs = row['match_idxs'][match_idxs]
    vals = row['METFRAG_BROKEN_BONDS'][bonds_idxs]
    np.put(broken_bonds, match_idxs, vals)
    return broken_bonds


def _decode(value):
    '''Decode.'''
    if isinstance(value, tuple):
        return tuple([_decode_val(val) for val in value])

    return value


def _decode_val(encoded):
    '''Decode value.'''
    table = Chem.GetPeriodicTable()

    atomic_number_1 = (encoded & 2**18 - 1) >> 11
    atomic_number_2 = (encoded & 2**11 - 1) >> 4
    order_ordinal = (encoded & 2**4 - 1) >> 1
    aromatic = (encoded & 1) == 1

    return table.GetElementSymbol(atomic_number_1), \
        table.GetElementSymbol(atomic_number_2), \
        order_ordinal + 1, \
        aromatic


def main(args):
    '''main method.'''
    df = get_data(args[0])
    bonds_frequency_df = get_bonds_frequency(df)
    bonds_frequency_df.to_csv('bonds_frequency.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
