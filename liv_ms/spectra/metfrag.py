'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
import re
import sys

from scipy.stats import norm

from liv_ms.plot import plot_spectrum
from liv_ms.spectra import searcher
from liv_ms.spectra.similarity import SimpleSpectraMatcher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def self_self(spec_df, match_func, min_mass=0.0, num_peaks=sys.maxsize):
    '''Compare measured spectra to MetFrag generated.'''

    _process_spec(spec_df)

    scores = []

    for _, row in spec_df.iterrows():
        meas_spec = list(zip(*row[['m/z', 'I']]))
        metfrag_spec = [[m, 100.0] for m in row['MetFrag m/z'] if m > min_mass]

        if meas_spec and metfrag_spec:
            meas_spec = sorted(meas_spec, key=lambda x: -x[1])[:num_peaks]
            meas_spec = sorted(meas_spec)

            # matcher = SimpleSpectraMatcher(np.array([meas_spec]), mass_acc,
            #                               scorer)

            matcher = match_func(np.array([meas_spec]))
            score = matcher.search(np.array([metfrag_spec]))[0][0]
            scores.append(score)

            if score < 0.01:
                plot_spectrum({'spectrum': meas_spec,
                               'name': row['smiles']},
                              [{'spectrum': metfrag_spec,
                                'name': row['smiles'],
                                'score': score}],
                              out_dir='bin_good')

    _plot_hist(scores, 'Similarity scores')


def search(spec_df, match_func, num_queries=32, num_hits=5):
    '''Search randomly selected MetFrag spectra against all.'''
    _process_spec(spec_df)

    spec_df.drop_duplicates('smiles', inplace=True)

    lib_df = spec_df.drop(['m/z', 'I'], axis=1).rename(
        columns={'MetFrag m/z': 'm/z', 'MetFrag I': 'I'})

    query_df = spec_df

    hits = searcher.random_search(match_func, lib_df, query_df,
                                  num_queries=num_queries, num_hits=num_hits,
                                  plot_dir='out/metfrag/search/' +
                                  match_func.func.__module__)

    print(hits)


def _plot_hist(values, label, out_filename='out.png'):
    '''Plot histogram.'''
    mu = np.mean(values)
    sigma = np.std(values)

    _, bins, _ = plt.hist(values, 50,
                          facecolor='green', alpha=0.75)

    best_fit_y = norm.pdf(bins, mu, sigma)

    plt.plot(bins, best_fit_y, 'r--', linewidth=1,
             label='mu=%.2f\nsigma=%.3f' % (mu, sigma))

    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title(label)
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig(out_filename)
    plt.close()


def _process_spec(spec_df):
    '''Process spectra data.'''
    spec_df['m/z'] = spec_df['m/z'].map(_to_array)
    spec_df['I'] = spec_df['I'].map(_to_array)
    spec_df['MetFrag m/z'] = spec_df['MetFrag m/z'].map(_to_array)
    spec_df['MetFrag I'] = spec_df['MetFrag m/z'].apply(_to_ones)


def _to_array(array_str):
    '''Parse array string.'''
    regex = re.compile('[\\s|,]+')
    terms = regex.split(array_str[1:-1].strip())
    return [float(val) for val in terms]


def _to_ones(array):
    '''Parse array string.'''
    return [100.0 for _ in array]


def main(args):
    '''main method.'''
    spec_df = pd.read_csv(args[0])

    match_func = partial(SimpleSpectraMatcher, mass_acc=0.005, scorer=np.min)
    # self_self(spec_df)
    search(spec_df, match_func, num_queries=128)


if __name__ == '__main__':
    # import sys;sys.argv = ['', 'Test.testName']
    main(sys.argv[1:])
