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
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.rdMolDescriptors import \
    GetHashedAtomPairFingerprintAsBitVect, \
    GetHashedTopologicalTorsionFingerprintAsBitVect
from sklearn.metrics.pairwise import cosine_similarity

from liv_ms import similarity
import numpy as np


def get_fngrprnt_funcs():
    '''Get fingerprint functions.'''
    fngrprnt_funcs = [
        None,
        GetHashedAtomPairFingerprintAsBitVect,
        GetHashedTopologicalTorsionFingerprintAsBitVect,
        GetAvalonFP,
        GetErGFingerprint
    ]

    for radius in range(2, 10):
        fngrprnt_funcs.append(partial(GetMorganFingerprintAsBitVect,
                                      radius=radius))

    for max_path in range(3, 10):
        fngrprnt_funcs.append(partial(Chem.RDKFingerprint,
                                      maxPath=max_path))

    return fngrprnt_funcs
    # return [None]


def encode_desc(smiles):
    '''Encode descriptors.'''
    mol = Chem.MolFromSmiles(smiles)
    return np.array([func(mol) for _, func in Descriptors.descList])


def encode_fngrprnt(smiles, fngrprnt_func):
    '''Encode fingerprints.'''
    if not fngrprnt_func:
        return []

    mol = Chem.MolFromSmiles(smiles)
    return fngrprnt_func(mol)


def get_similarities(smiles, fngrprnt_func):
    '''Get similarities between chemicals represented by SMILES.'''
    dists = {}

    fps = [encode_fngrprnt(sml, fngrprnt_func) for sml in smiles]

    for idx1, fp1 in enumerate(fps):
        for idx2 in range(idx1, len(fps)):
            if isinstance(fp1, DataStructs.cDataStructs.ExplicitBitVect):
                sim = DataStructs.FingerprintSimilarity(fp1, fps[idx2])
            else:
                sim = cosine_similarity(fp1.reshape(1, -1),
                                        fps[idx2].reshape(1, -1))[0][0]

            dists[(smiles[idx1], smiles[idx2])] = 1 - sim

    return dists


def main(args):
    '''main method.'''
    print(get_similarities(args, Chem.RDKFingerprint))


if __name__ == '__main__':
    main(sys.argv[1:])
