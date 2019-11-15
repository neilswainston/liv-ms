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
from liv_ms.searcher import SpectraSearcher
from liv_ms.spectra import mona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_queries(query_names, query_specs, lib_df, lib_specs, num_hits):
    '''Run queries.'''
    # Initialise SpectraMatcher:
    matcher = similarity.SimpleSpectraMatcher(lib_specs)
    src = SpectraSearcher(matcher, lib_df)

    # Run queries:
    hits = src.search(query_specs, num_hits)

    print(hits)

    # Plot results:
    hit_specs = lib_specs.take([[val['index'] for val in hit] for hit in hits])
    plot_spectra(query_names, query_specs, hits, hit_specs)


def plot_spectra(query_names, queries, hit_data, hit_specs):
    '''Plot spectra.'''
    for query_name, query_spec, hit, hit_spec in zip(query_names, queries,
                                                     hit_data, hit_specs):
        query = {'name': query_name, 'spectrum': query_spec}

        for h, s in zip(hit, hit_spec):
            h.update({'spectrum': s})

        plot.plot_spectrum(query, hit)


def _random_search(lib_df):
    '''Random search.'''
    lib_specs = spectra.get_spectra(lib_df)

    # Get queries:
    num_queries = 16
    query_df = lib_df.sample(num_queries)
    query_specs = spectra.get_spectra(query_df)

    # Run queries:
    num_hits = 5
    run_queries(query_df['name'], query_specs, lib_df, lib_specs, num_hits)


def _specific_search(lib_df, query_idx, lib_idx):
    '''Specific search.'''
    lib_specs = spectra.get_spectra(lib_df.loc[[lib_idx]])

    # Get queries:
    query_df = lib_df.loc[[query_idx]]
    query_specs = spectra.get_spectra(query_df)

    # Run queries:
    num_hits = 1
    run_queries(query_df['name'], query_specs, lib_df, lib_specs, num_hits)


def main(args):
    '''main method.'''

    # Get spectra:
    num_spectra = 256
    df = mona.get_spectra(args[0], num_spectra)

    _random_search(df)
    # _specific_search(df, 125, 19)


if __name__ == '__main__':
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

    main(sys.argv[1:])

    pr.disable()

    pr.print_stats(sort='time')
