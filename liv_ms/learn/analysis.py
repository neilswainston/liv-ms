'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
from ast import literal_eval as make_tuple
import collections
from functools import partial
import json
import sys

from rdkit import Chem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data(filename, tol):
    '''Get data.'''
    df = pd.read_csv(filename)
    df['m/z'] = df['m/z'].apply(_to_numpy, sep=',')
    df['METFRAG_MZ'] = df['METFRAG_MZ'].apply(_to_numpy, sep=',')
    df['METFRAG_BROKEN_BONDS'] = df['METFRAG_BROKEN_BONDS'].apply(_to_numpy_2d)

    df['match_idxs'] = df.apply(partial(_match, tol=tol), axis=1)
    df['bonds_broken'] = df.apply(_get_broken_bonds, axis=1)

    return df


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


def plot(bond_freq_df, out_filename):
    '''Plot.'''
    categories, labels, data = _get_plot_data(bond_freq_df)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects = []

    for idx, vals in enumerate(data):
        rects.append(ax.bar(x + width * idx, vals, width,
                            label=categories[idx]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    ax.set_title('Frequencies of broken bonds of matching fragments')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects:
        _autolabel(ax, rect)

    fig.tight_layout()

    plt.savefig(out_filename)


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

    if row['METFRAG_MZ'].size:
        abs_diff = np.abs(np.subtract.outer(
            row['m/z'], row['METFRAG_MZ'])) < tol
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


def _get_plot_data(df, min_freq=0.001):
    '''Get plot data.'''
    match_df = df[(df['match']) & (df['freq_matched'] > min_freq)]
    match_df.loc[:, 'bond_label'] = match_df.apply(_get_bond_label, axis=1)

    categories = match_df['aromatic'].unique()
    labels = match_df['bond_label'].unique()
    data = np.zeros((len(categories), len(labels)))

    match_df.apply(partial(_add_plot_data,
                           categories=categories,
                           labels=labels,
                           data=data), axis=1)

    return ['Aromatic' if cat else 'Non-aromatic' for cat in categories], \
        labels, data


def _get_bond_label(row):
    '''Get bond label.'''
    bond_chrs = {1.0: '-',
                 2.0: '=',
                 3.0: '#',
                 4.0: '$'}

    try:
        return row['atom1'] + bond_chrs[row['order']] + row['atom2']
    except KeyError:
        return 'PREC'


def _add_plot_data(row, categories, labels, data):
    '''Add plot data.'''
    category = np.argwhere(categories == row['aromatic'])[0]
    label = np.argwhere(labels == row['bond_label'])[0]
    data[category, label] = row['freq_matched']


def _autolabel(ax, rects):
    '''Attach a text label above each bar in *rects*, displaying its height.'''
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height if height else '',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')


def _make_tuple(val):
    '''Make tuple.'''
    if isinstance(val, float) and np.isnan(val):
        return val

    return make_tuple(val)


def main(args):
    '''main method.'''
    in_filename = args[0]

    if in_filename.startswith('bonds_freq'):
        bonds_freq_df = pd.read_csv(in_filename)
        bonds_freq_df['bonds'] = bonds_freq_df['bonds'].apply(_make_tuple)
        bonds_freq_df.set_index('bonds', inplace=True)
    else:
        df = get_data(in_filename, float(args[1]))
        bonds_freq_df = get_bonds_freq(df)
        bonds_freq_df.to_csv('bonds_freq.csv')

    bond_freq_df = get_bond_freq(bonds_freq_df)
    bond_freq_df.to_csv('bond_freq.csv', index=False)

    plot(bond_freq_df, in_filename + '.png')


if __name__ == '__main__':
    main(sys.argv[1:])
