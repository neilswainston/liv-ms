'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
from functools import partial
import inspect
from itertools import product, zip_longest
import os.path
import random
import sys

from rdkit import Chem
from rdkit.Chem import Draw

from liv_ms import chem, plot, searcher, similarity, data, utils
from liv_ms.data import mona, spectra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyse(df, fngrprnt_func, match_func, out_dir):
    '''Analyse correlation between spectra match score and chemical
    similarity.'''
    hits = random_search(match_func, df)
    # specific_search(matcher, df, 125, 19)

    hit_results = []

    for hit in hits:
        for h in hit[1:]:
            smiles = (hit[0]['smiles'], h['smiles'])
            chem_sim = chem.get_similarities(smiles, fngrprnt_func)

            hit_results.append([hit[0]['name'], hit[0]['smiles'],
                                h['name'], h['smiles'],
                                h['score'], chem_sim[smiles]])

    hit_df = pd.DataFrame(hit_results, columns=['query_name',
                                                'query_smiles',
                                                'hit_name',
                                                'hit_smiles',
                                                'score',
                                                'chem_sim'])

    name = '%s, %s' % (utils.to_str(fngrprnt_func),
                       utils.to_str(match_func))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hit_df.to_csv(os.path.join(out_dir, '%s.csv' % name),
                  index=False)

    plot.plot_scatter(
        hit_df['score'], hit_df['chem_sim'],
        name, 'spec_sim', 'chem_sim', out_dir='out')


def random_search(match_func, lib_df, num_queries=32, num_hits=64,
                  plot_dir=None):
    '''Random search.'''
    lib_specs = spectra.get_spectra(lib_df)

    # Get queries:
    query_df = lib_df.sample(num_queries)
    query_specs = spectra.get_spectra(query_df)

    # Run queries:
    return run_queries(match_func, query_df['name'], query_specs,
                       lib_df, lib_specs,
                       num_hits=num_hits, plot_dir=plot_dir)


def specific_search(match_func, lib_df, query_idx, lib_idx, plot_dir=None):
    '''Specific search.'''
    lib_specs = spectra.get_spectra(lib_df.loc[[lib_idx]])

    # Get queries:
    query_df = lib_df.loc[[query_idx]]
    query_specs = spectra.get_spectra(query_df)

    # Run queries:
    return run_queries(match_func, query_df['name'], query_specs,
                       lib_df, lib_specs,
                       num_hits=1, plot_dir=plot_dir)


def run_queries(match_func, query_names, query_specs, lib_df, lib_specs,
                num_hits, plot_dir=None):
    '''Run queries.'''
    # Initialise SpectraMatcher:
    src = searcher.SpectraSearcher(match_func(lib_specs), lib_df)

    # Run queries:
    hits = src.search(query_specs, num_hits)

    # Plot results:
    if plot_dir:
        hit_specs = lib_specs.take(
            [[val['index'] for val in hit] for hit in hits])
        _plot_spectra(query_names, query_specs, hits, hit_specs,
                      out_dir=plot_dir)

    return hits


def _plot_spectra(query_names, queries, hit_data, hit_specs, out_dir):
    '''Plot spectra.'''
    for query_name, query_spec, hit, hit_spec in zip(query_names, queries,
                                                     hit_data, hit_specs):
        query = {'name': query_name, 'spectrum': query_spec}

        for h, s in zip(hit, hit_spec):
            h.update({'spectrum': s})

        plot.plot_spectrum(query, hit, out_dir)


def _get_match_funcs():
    '''Get match functions.'''
    match_funcs = []

    for mass_acc in [0.001, 0.003, 0.01, 0.03, 0.1]:
        for scorer in [np.max, np.average]:
            match_funcs.append(partial(similarity.SimpleSpectraMatcher,
                                       mass_acc=mass_acc,
                                       scorer=scorer))

    return match_funcs


def main(args):
    '''main method.'''
    out_dir = args[1]

    # Get spectra:
    df = mona.get_spectra(args[0], num_spec=1024)

    for fngrprnt_func, match_func in product(chem.get_fngrprnt_funcs(),
                                             _get_match_funcs()):
        if fngrprnt_func:
            analyse(df, fngrprnt_func, match_func, out_dir)


if __name__ == '__main__':
    # import cProfile

    # pr = cProfile.Profile()
    # pr.enable()

    main(sys.argv[1:])

    # pr.disable()

    # pr.print_stats(sort='time')
