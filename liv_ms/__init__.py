'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
from itertools import zip_longest
import os.path
import random
import sys

from rdkit import Chem
from rdkit.Chem import Draw

from liv_ms import plot, similarity, spectra
from liv_ms.spectra import mona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_queries(query_names, query_specs, lib_specs, df, num_hits):
    '''Run queries.'''
    # Initialise SpectraMatcher:
    matcher = similarity.SimpleSpectraMatcher(lib_specs)

    # Run queries:
    result = search(matcher, query_specs, df, num_hits)

    print(result)

    # Plot results:
    plot_spectra(query_names, query_specs, df, result)


def search(matcher, query_spec, df, num_hits):
    '''Search.'''
    import time
    start = time.time()

    # Search:
    res = matcher.search(query_spec)

    # Get indexes of top n hits:
    fnc = partial(_get_top_idxs, n=num_hits)
    top_idxs = np.apply_along_axis(fnc, 1, res)

    # Get score data corresponding to top n hits:
    offset = np.arange(0, res.size, res.shape[1])
    score_data = np.take(res, offset[:, np.newaxis] + top_idxs)

    # Get match data corresponding to top n hits:
    df.reset_index(inplace=True)
    fnc = partial(_get_data, data=df[['index',
                                      'name',
                                      'monoisotopic_mass_float',
                                      'smiles']])

    match_data = np.apply_along_axis(fnc, 1, top_idxs)

    print(time.time() - start)

    return np.dstack((match_data, score_data))


def plot_spectra(query_names, queries, df, results):
    '''Plot spectra.'''
    for query_name, query_spec, result in zip(query_names, queries, results):
        query = {'name': query_name, 'spectrum': query_spec}

        hits = zip(*[result[:, 1], result[:, 4],
                     spectra.get_spectra(df.loc[result[:, 0]])])

        hits = [dict(zip(['name', 'score', 'spectrum'], hit)) for hit in hits]

        plot.plot_spectrum(query, hits)


def _get_top_idxs(arr, n):
    '''Get sorted list of top indices.'''
    idxs = np.argpartition(arr, n - 1)[:n]

    # Extra code if you need the indices in order:
    min_elements = arr[idxs]
    min_elements_order = np.argsort(min_elements)

    return idxs[min_elements_order]


def _get_data(idxs, data):
    '''Get data for best matches.'''
    return data.loc[idxs]


def main(args):
    '''main method.'''
    num_spectra = 256
    num_queries = 16
    num_hits = 5

    # Get spectra:
    df = mona.get_spectra(args[0], num_spectra)
    lib_specs = spectra.get_spectra(df)

    # Get queries:
    query_df = df.sample(num_queries)
    query_specs = spectra.get_spectra(query_df)

    # Run queries:
    run_queries(query_df['name'], query_specs, lib_specs, df, num_hits)


if __name__ == '__main__':
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

    main(sys.argv[1:])

    pr.disable()

    pr.print_stats(sort='time')
