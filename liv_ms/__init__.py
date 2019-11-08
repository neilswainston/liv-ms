'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=ungrouped-imports
# pylint: disable=wrong-import-order
from functools import partial
from itertools import zip_longest
import os.path
import random
import sys

from matplotlib import collections
from rdkit import Chem
from rdkit.Chem import Draw

from liv_ms import similarity, spectra
from liv_ms.spectra import mona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_spectra(query_df, df, results):
    '''Plot spectra.'''
    for (_, query), result in zip(query_df.iterrows(), results):
        _plot_spectrum(query, df, result)


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


def _plot_spectrum(query, df, results, out_dir='out'):
    '''Plot spectrum.'''
    # Get data:
    query_spec = spectra.get_spectra(query.to_frame().T)[0]
    query_lines = [[(x, 0), (x, y)] for x, y in query_spec]
    query_col = ['green' for _ in query_spec]

    lib_specs = spectra.get_spectra(df.loc[results[:, 0]])

    # img = Draw.MolToImage(Chem.MolFromSmiles(res[3]), bgColor=None)

    # Make plot
    fig, axes = plt.subplots(len(results), 1, sharex=True)

    for ax, res, lib_spec in zip(axes, results, lib_specs):
        ax.axhline(y=0, color='k', linewidth=1)
        ax.margins(x=0, y=0)

        # Add 'peaks':
        ax.add_collection(
            collections.LineCollection(
                query_lines + [[(x, 0), (x, -y)] for x, y in lib_spec],
                colors=query_col + ['red' for _ in lib_spec],
                alpha=0.5))

        # Add (invisible) scatter points:
        ax.scatter(*zip(*query_spec), s=0)
        ax.scatter(*zip(*lib_spec), s=0)

        # Format and save:
        name = '_'.join([query['name'], res[1], '%.3f' % res[4]])
        ax.set_title(name, fontsize=6)
        ax.set_xlabel('m/z', fontsize=6)
        ax.set_ylabel('I', fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=4)
        # ax2.figimage(img, 0, fig.bbox.ymax - img.size[1])

    fig.tight_layout()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, query['name'] + '.png'), dpi=800)


def main(args):
    '''main method.'''
    num_spectra = 256
    num_queries = 16
    num_hits = 5

    # Get spectra:
    df = mona.get_spectra(args[0], num_spectra)
    specs = spectra.get_spectra(df)

    # Initialise SpectraMatcher:
    # matcher = similarity.KDTreeSpectraMatcher(specs, use_i=False)
    matcher = similarity.SimpleSpectraMatcher(specs)

    # Run queries:
    query_df = df.sample(num_queries)
    queries = spectra.get_spectra(query_df)
    result = search(matcher, queries, df, num_hits)

    # Plot results:
    plot_spectra(query_df, df, result)

    print(query_df['name'])
    print(result)


if __name__ == '__main__':
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

    main(sys.argv[1:])

    pr.disable()

    pr.print_stats(sort='time')
