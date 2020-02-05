'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from liv_ms.data import rt
import pandas as pd


def get_rt_data(filename, num_spec=1e32, regen_stats=False):
    '''Get RT data.'''
    # Get spectra:
    df = pd.read_csv(filename)

    # Drop missing SMILES:
    df.rename(columns={'shikifactory compounds': 'name',
                       'Canonical SMILES': 'smiles'}, inplace=True)
    df.dropna(subset=['smiles'], inplace=True)

    # Add values:
    df['column values'] = [[2.1, 50.0, 1.8, 1.0] for _ in df.index]
    df['flow rate values'] = [[0.1] * 60 for _ in df.index]

    grad_terms = [[0.0, 0.0], [1.5, 0.0], [7.0, 1.0], [8.5, 1.0], [8.51, 0.0]]
    grad_vals = rt.get_timecourse_vals(list(zip(*grad_terms)))
    df['gradient values'] = [grad_vals for _ in df.index]

    return rt.get_stats(df)
