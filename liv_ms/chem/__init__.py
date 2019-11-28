'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=wrong-import-order
from functools import partial
import sys

from rdkit import Chem, DataStructs
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.rdMolDescriptors import \
    GetHashedAtomPairFingerprintAsBitVect, \
    GetHashedTopologicalTorsionFingerprintAsBitVect

from liv_ms import similarity
import numpy as np


def get_fngrprnt_funcs():
    '''Get fingerprint functions.'''
    fngrprnt_funcs = []

    fngrprnt_funcs.append(GetHashedAtomPairFingerprintAsBitVect)
    fngrprnt_funcs.append(GetHashedTopologicalTorsionFingerprintAsBitVect)
    fngrprnt_funcs.append(GetAvalonFP)
    fngrprnt_funcs.append(GetErGFingerprint)

    for radius in range(2, 10):
        fngrprnt_funcs.append(partial(GetMorganFingerprintAsBitVect,
                                      radius=radius))

    for max_path in range(3, 10):
        fngrprnt_funcs.append(partial(Chem.RDKFingerprint,
                                      maxPath=max_path))

    return fngrprnt_funcs


def encode(smiles, fngrprnt_func):
    '''Encode SMILES.'''
    mol = Chem.MolFromSmiles(smiles)
    return np.array(fngrprnt_func(mol))


def get_similarities(smiles, fngrprnt_func):
    '''Get similarities between chemicals represented by SMILES.'''
    sims = {}

    fps = [encode(sml, fngrprnt_func) for sml in smiles]

    for idx1, fp1 in enumerate(fps):
        for idx2 in range(idx1, len(fps)):
            sims[(smiles[idx1], smiles[idx2])] = \
                1 - DataStructs.FingerprintSimilarity(fp1, fps[idx2])

    return sims


def main(args):
    '''main method.'''
    print(get_similarities(args, Chem.RDKFingerprint))


if __name__ == '__main__':
    main(sys.argv[1:])
