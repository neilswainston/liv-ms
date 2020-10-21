'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
from functools import partial
from itertools import product
import os.path
import sys

from liv_ms import chem, data, plot, to_str
from liv_ms.spectra import searcher, similarity
import numpy as np
import pandas as pd


def analyse(df, fngrprnt_func, match_func, out_dir, num_queries=8):
    '''Analyse correlation between spectra match score and chemical
    similarity.'''
    hits = searcher.random_mass_search(match_func, df, num_queries)
    # hits = searcher.random_search(match_func, df, num_queries)
    # hits = searcher.specific_search(matcher, df, [125], [19])

    _process_hits(hits, fngrprnt_func, match_func, out_dir)


def _process_hits(hits, fngrprnt_func, match_func, out_dir):
    '''Process hits.'''
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

    name = '%s, %s' % (to_str(fngrprnt_func), to_str(match_func))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hit_df.to_csv(os.path.join(out_dir, '%s.csv' % name),
                  index=False)

    plot.plot_scatter(
        hit_df['score'], hit_df['chem_sim'],
        name, 'spec_sim', 'chem_sim', out_dir='out')


def _get_match_funcs():
    '''Get match functions.'''
    match_funcs = []

    for mass_acc in [0.001, 0.003, 0.01, 0.03, 0.1]:
        for scorer in [np.max, np.average]:
            match_funcs.append(partial(similarity.SimpleSpectraMatcher,
                                       mass_acc=mass_acc,
                                       scorer=scorer))

    return match_funcs


def _plot_spectra(query_names, queries, hits, hit_data, lib_specs, lib_df,
                  out_dir):
    '''Plot spectra.'''
    hit_specs = lib_specs.take(
        [[lib_df.index.get_loc(val['index']) for val in hit]
         for hit in hits])

    for query_name, query_spec, hit, hit_spec in zip(query_names, queries,
                                                     hit_data, hit_specs):
        query = {'name': query_name, 'spectrum': query_spec}

        for h, s in zip(hit, hit_spec):
            h.update({'spectrum': s})

        plot.plot_spectrum(query, hit, out_dir)


def main(args):
    '''main method.'''

    # Get spectra:
    df = pd.read_csv(args[0], nrows=512)
    df['m/z'] = df['m/z'].apply(data.to_numpy)
    df['I'] = df['I'].apply(data.to_numpy)

    out_dir = args[1]

    for fngrprnt_func, match_func in product(chem.get_fngrprnt_funcs(),
                                             _get_match_funcs()):
        if fngrprnt_func:
            analyse(df, fngrprnt_func, match_func, out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
