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
import random
import sys

from matplotlib import collections
from rdkit import Chem
from rdkit.Chem import Draw

from liv_ms import similarity
from liv_ms.spectra import mona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_df(chem, spec):
    '''Get DataFrame.'''
    chem_df = pd.DataFrame(chem)
    spec_df = pd.DataFrame(spec)

    return chem_df.join(spec_df)


def _search(matcher, query_spec, df, num_hits):
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


def _get_top_idxs(arr, n):
    '''Get sorted list of top indices.'''
    idxs = np.argpartition(arr, n - 1)[:n]

    # Extra code if you need the indices in order:
    min_elements = arr[idxs]
    min_elements_order = np.argsort(min_elements)
    return idxs[min_elements_order]


def _get_spectra(df):
    '''Ger spectra.'''
    spectra = df.apply(_get_peaks, axis=1)
    return np.array(spectra.tolist())


def _get_peaks(row):
    '''Get peaks.'''
    return np.column_stack(row[['m/z', 'I']])


def _get_data(idxs, data):
    '''Get data for best matches.'''
    return data.loc[idxs]


def _plot_spectra(query_df, df, results):
    '''Plot spectra.'''
    for (_, query), result in zip(query_df.iterrows(), results):
        for res in result:
            _plot_spectrum(query, df, res)


def _plot_spectrum(query, df, res):
    '''Plot spectrum.'''
    # Get data:
    query_spec = _get_spectra(query.to_frame().T)[0]
    lib_spec = _get_spectra(df.loc[[res[0]]])[0]
    name = '_'.join([query['name'], res[1], '%.3f' % res[4]])

    # img = Draw.MolToImage(Chem.MolFromSmiles(res[3]), bgColor=None)

    # Make plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.margins(x=0, y=0)

    # Add 'peaks':
    query_lines = [[(x, 0), (x, y)] for x, y in query_spec]
    lib_lines = [[(x, 0), (x, -y)] for x, y in lib_spec]
    query_col = ['green' for _ in query_spec]
    lib_col = ['red' for _ in lib_spec]

    ax1.add_collection(collections.LineCollection(query_lines + lib_lines,
                                                  colors=query_col + lib_col,
                                                  alpha=0.5))

    # Add (invisible) scatter points:
    ax1.scatter(*zip(*query_spec), s=0)
    ax1.scatter(*zip(*lib_spec), s=0)

    # Format and save:
    ax1.set_title(name)
    # ax2.figimage(img, 0, fig.bbox.ymax - img.size[1])

    plt.savefig(name + '.png')


def main(args):
    '''main method.'''
    num_spectra = 16
    num_queries = 4
    num_hits = 3

    chem, spec = mona.get_spectra(args[0], num_spectra)
    df = get_df(chem, spec)

    spectra = _get_spectra(df)
    matcher = similarity.SpectraMatcher(spectra)

    query_df = df.sample(num_queries)
    queries = _get_spectra(query_df)

    result = _search(matcher, queries, df, num_hits)

    _plot_spectra(query_df, df, result)

    print(query_df['name'])
    print(result)


if __name__ == '__main__':
    main(sys.argv[1:])
