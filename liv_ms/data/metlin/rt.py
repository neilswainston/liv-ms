'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=broad-except
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
from rdkit.Chem import inchi, rdmolfiles
from liv_ms.data import rt
import pandas as pd


def get_rt_data(filename, num_spec=1e32):
    '''Get RT data.'''
    # Get spectra:
    df = pd.read_csv(filename, sep=';', nrows=num_spec)

    # Convert RT to minutes and rename column:
    df['retention time'] = df['rt'] / 60.0

    # Convert InChI to SMILES:
    df['smiles'] = df['inchi'].apply(_get_smiles)
    df.dropna(subset=['smiles'], inplace=True)

    # Add values:
    df['column values'] = [[2.1, 50.0, 1.8, 1.0] for _ in df.index]
    df['flow rate values'] = [[0.1] * 60 for _ in df.index]

    grad_terms = [[0.0, 0.05], [3.0, 0.05],
                  [5.0, 0.5], [15.0, 0.85], [18.0, 0.85]]
    grad_vals = rt.get_timecourse_vals(list(zip(*grad_terms)))
    df['gradient values'] = [grad_vals for _ in df.index]

    return rt.get_stats(df)


def _get_smiles(inchi_term):
    '''Get smiles.'''
    try:
        mol = inchi.MolFromInchi(inchi_term, treatWarningAsError=True)
        return rdmolfiles.MolToSmiles(mol)
    except Exception:
        return None

# get_rt_data('data/SMRT_dataset.csv',
#            num_spec=10).to_csv('out.csv', index=False)
