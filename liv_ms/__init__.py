'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from functools import partial
import random
import sys

from sympy.polys.polyconfig import query

from liv_ms import similarity
from liv_ms.spectra import mona
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
    top_idxs = np.apply_along_axis(fnc, 1, -res)

    # Get score data corresponding to top n hits:
    offset = np.arange(0, res.size, res.shape[1])
    score_data = np.take(res, offset[:, np.newaxis] + top_idxs)

    # Get match data corresponding to top n hits:
    fnc = partial(_get_data, data=df[['name', 'smiles']])
    match_data = np.apply_along_axis(fnc, 1, top_idxs)

    print(time.time() - start)

    return score_data, match_data


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
    num_spectra = 6000
    num_queries = 16
    num_hits = 10

    chem, spec = mona.get_spectra(args[0], num_spectra)  # int(args[1]))
    df = get_df(chem, spec)

    spectra = df[['m/z', 'I']].values

    matcher = similarity.SpectraMatcher(spectra)

    query_df = df.sample(num_queries)
    query_spec = query_df[['m/z', 'I']].values

    score_data, match_data = _search(matcher, query_spec, df, num_hits)

    print(query_df['name'])
    print(score_data)
    print(match_data)


if __name__ == '__main__':
    main(sys.argv[1:])
