'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
import re
import sys

from scipy.stats import norm

from liv_ms.spectra.similarity import SimpleSpectraMatcher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare(spec_df, min_mass=0.0, mass_acc=0.01, num_peaks=10,
            scorer=np.min):
    '''Compare measured spectra to MetFrag generated.'''

    spec_df['m/z'] = spec_df['m/z'].map(_to_array)
    spec_df['I'] = spec_df['I'].map(_to_array)
    spec_df['MetFrag m/z'] = spec_df['MetFrag m/z'].map(_to_array)

    scores = []

    for _, row in spec_df.iterrows():
        meas_spec = list(zip(*row[['m/z', 'I']]))
        query_spec = [[m, 1] for m in row['MetFrag m/z'] if m > min_mass]

        if meas_spec and query_spec:
            meas_spec = sorted(meas_spec, key=lambda x: -x[1])[:num_peaks]
            meas_spec = sorted(meas_spec)

            print(meas_spec)
            matcher = SimpleSpectraMatcher(np.array([meas_spec]), mass_acc,
                                           scorer)
            score = matcher.search(np.array([query_spec]))[0][0]
            scores.append(score)

        # print(row['smiles'], score)

    _plot_hist(scores, 'Similarity scores')


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


def _to_array(array_str):
    '''Parse array string.'''
    regex = re.compile('[\\s|,]+')
    terms = regex.split(array_str[1:-1].strip())
    return [float(val) for val in terms]


def main(args):
    '''main method.'''
    spec_df = pd.read_csv(args[0])
    compare(spec_df)


if __name__ == '__main__':
    # import sys;sys.argv = ['', 'Test.testName']
    main(sys.argv[1:])
