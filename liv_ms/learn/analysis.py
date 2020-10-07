'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
import collections
from functools import partial
import json
import sys

from rdkit import Chem

import numpy as np
import pandas as pd


def get_bonds_freq(df):
    '''Get multiple bonds frequency.'''
    bonds_freq = pd.Series(
        [item for sublist in df['bonds_broken']
         for item in sublist]).value_counts(normalize=True, dropna=False)

    bonds_freq_df = bonds_freq.reset_index()
    bonds_freq_df['index'] = bonds_freq_df['index'].apply(_decode)
    bonds_freq_df.set_index('index', inplace=True)
    bonds_freq_df.index.name = 'bonds'
    bonds_freq_df.columns = ['freq']

    return bonds_freq_df


def get_bond_freq(bonds_freq_df):
    '''Get individual bond frequency.'''
    bond_freq = collections.defaultdict(int)

    partial_bond_freq = partial(_get_bond_freq, bond_freq=bond_freq)
    bonds_freq_df.apply(partial_bond_freq, axis=1)

    data = []

    for key, value in bond_freq.items():
        data.append(list(key) + [value])

    cols = ['atom1', 'atom2', 'order', 'aromatic', 'match', 'precursor',
            'freq']

    df = pd.DataFrame(data, columns=cols)

    matched_freq = df.loc[df['match'], 'freq']
    df.loc[df['match'], 'freq_matched'] = matched_freq / sum(matched_freq)

    return df.sort_values('freq', ascending=False)


def get_data(filename, tol):
    '''Get data.'''
    df = pd.read_csv(filename)
    df['m/z'] = df['m/z'].apply(_to_numpy, sep=',')
    df['METFRAG_MZ'] = df['METFRAG_MZ'].apply(_to_numpy, sep=',')
    df['METFRAG_BROKEN_BONDS'] = df['METFRAG_BROKEN_BONDS'].apply(_to_numpy_2d)

    df['match_idxs'] = df.apply(partial(_match, tol=tol), axis=1)
    df['bonds_broken'] = df.apply(_get_broken_bonds, axis=1)

    return df


def _to_numpy(array_str, sep=','):
    '''Convert array_str to numpy.'''
    return np.fromstring(array_str[1:-1], sep=sep)


def _to_numpy_2d(array_str):
    '''Convert array_str to numpy 2d array.'''
    values = json.loads(array_str)
    return np.array([tuple(val) for val in values])


def _match(row, tol):
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


def _get_bond_freq(row, bond_freq):
    '''Get individual bond frequency for given row.'''
    if isinstance(row.name, float) and np.isnan(row.name):
        # Special case: unmatched:
        bond_freq[
            (None, None, float('NaN'), False, False, False)] += row.freq
    elif not row.name:
        # Special case: precursor:
        bond_freq[
            (None, None, float('NaN'), False, True, True)] += row.freq
    else:
        for bond in row.name:
            key = tuple(list(bond) + [True, False])
            bond_freq[key] += row.freq / len(row.name)


def main(args):
    '''main method.'''
    df = get_data(args[0], float(args[1]))
    bonds_freq_df = get_bonds_freq(df)
    bonds_freq_df.to_csv('bonds_freq.csv')

    bond_freq_df = get_bond_freq(bonds_freq_df)
    bond_freq_df.to_csv('bond_freq.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
