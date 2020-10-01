'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
import json
import sys

import numpy as np
import pandas as pd


def to_numpy(array_str, sep=','):
    '''Convert array_str to numpy.'''
    return np.fromstring(array_str[1:-1], sep=sep)


def to_numpy_2d(array_str):
    '''Convert array_str to numpy 2d array.'''
    values = json.loads(array_str)
    return np.array([np.array(val) for val in values])


def match(row, tol=0.01):
    '''Determine if masses match.'''
    match_idxs = np.ones(row['m/z'].size, dtype=int) * -1
    abs_diff = np.abs(np.subtract.outer(row['m/z'], row['METFRAG_MZ'])) < tol
    abs_diff_match_idxs = np.where(np.any(abs_diff, axis=1))
    abs_diff_idxs = np.argmax(abs_diff, axis=1)[abs_diff_match_idxs]
    np.put(match_idxs, abs_diff_match_idxs, abs_diff_idxs, mode='raise')
    return match_idxs


def get_broken_bonds(row):
    '''Get bonds broken.'''
    broken_bonds = np.empty(row['match_idxs'].size, dtype=object)
    match_idxs = np.where(row['match_idxs'] > -1)
    bonds_idxs = row['match_idxs'][match_idxs]
    vals = row['METFRAG_BROKEN_BONDS'][bonds_idxs]
    np.put(broken_bonds, match_idxs, vals)
    return broken_bonds


def main(args):
    '''main method.'''
    df = pd.read_csv(args[0])
    df['m/z'] = df['m/z'].apply(to_numpy, sep=',')
    df['METFRAG_MZ'] = df['METFRAG_MZ'].apply(to_numpy, sep=',')
    df['METFRAG_BROKEN_BONDS'] = df['METFRAG_BROKEN_BONDS'].apply(to_numpy_2d)

    df['match_idxs'] = df.apply(match, axis=1)
    df['bonds_broken'] = df.apply(get_broken_bonds, axis=1)

    print(df[['match_idxs', 'bonds_broken']])


if __name__ == '__main__':
    main(sys.argv[1:])
