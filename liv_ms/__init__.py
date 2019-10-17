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


def main(args):
    '''main method.'''
    num_spectra = 10000
    num_queries = 100
    num_hits = 5

    chem, spec = mona.get_spectra(args[0], num_spectra)  # int(args[1]))
    df = get_df(chem, spec)

    spectra = df[['m/z', 'I']].values

    matcher = similarity.SpectraMatcher(spectra)

    query_df = df.sample(num_queries)
    print(query_df['name'])
    query_spec = query_df[['m/z', 'I']].values

    import time
    start = time.time()

    res = matcher.search(query_spec)

    score_data = -np.partition(-res, 1)

    fnc = partial(_get_data, n=num_hits,
                  data=df[['name', 'formula', 'smiles']])
    match_data = np.apply_along_axis(fnc, 1, np.argpartition(-res, 1))

    print(time.time() - start)
    print(score_data)
    print(match_data)


def _get_data(row, n, data):
    '''Get data for best matches.'''
    idxs = row[:n]
    return data.loc[idxs]


if __name__ == '__main__':
    main(sys.argv[1:])
