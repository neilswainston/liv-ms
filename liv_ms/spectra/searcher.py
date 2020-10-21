'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods
# pylint: disable=wrong-import-order
from functools import partial
from liv_ms.spectra import get_spectra
import numpy as np


class SpectraSearcher():
    '''Class to represent a SpectraSearcher.'''

    def __init__(self, matcher, lib_df):
        self.__matcher = matcher
        self.__lib_df = lib_df.reset_index()

    def search(self, query_specs, num_hits):
        '''Search.'''
        idx_score = self.__search(query_specs, num_hits)

        # Get match data corresponding to top n hits:
        fnc = partial(_get_data, data=self.__lib_df[['index',
                                                     'name',
                                                     'monoisotopic_mass_float',
                                                     'smiles']])

        match_data = np.apply_along_axis(fnc, 1, idx_score[:, :, 0])

        hits = np.dstack((match_data, idx_score[:, :, 1]))

        fnc = partial(_to_dict, keys=[
            'index', 'name', 'monoisotopic_mass_float', 'smiles',
            'score'])

        return np.apply_along_axis(fnc, 2, hits)

    def __search(self, query_spec, num_hits):
        '''Search.'''

        # Search:
        res = self.__matcher.search(query_spec)

        # Get indexes of top n hits:
        fnc = partial(_get_top_idxs, n=num_hits)
        top_idxs = np.apply_along_axis(fnc, 1, res)

        # Get score data corresponding to top n hits:
        offset = np.arange(0, res.size, res.shape[1])
        score_data = np.take(res, offset[:, np.newaxis] + top_idxs)

        return np.dstack((top_idxs, score_data))


def random_search(match_func, df, num_queries=32, num_hits=64):
    '''Random search.'''
    lib_idx = df.index

    # Get queries:
    query_idx = df.sample(num_queries).index

    # Run queries:
    return specific_search(match_func, df, query_idx, lib_idx, num_hits)


def specific_search(match_func, df, query_idx, lib_idx, num_hits=64):
    '''Specific search.'''
    lib_df = df.loc[lib_idx]
    lib_specs = get_spectra(lib_df)

    # Get queries:
    query_df = df.loc[query_idx]
    query_specs = get_spectra(query_df)

    # Run queries:
    return _run_queries(match_func, query_specs, lib_df, lib_specs, num_hits)


def _run_queries(match_func, query_specs, lib_df, lib_specs, num_hits):
    '''Run queries.'''
    # Initialise SpectraMatcher:
    src = SpectraSearcher(match_func(lib_specs), lib_df)

    # Run queries:
    return src.search(query_specs, num_hits)


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


def _to_dict(vals, keys):
    '''Convert to dictionary.'''
    return dict(zip(*[keys, vals]))
