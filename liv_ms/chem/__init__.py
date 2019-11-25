'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-member
import sys

from rdkit import Chem, DataStructs

from liv_ms import similarity
import numpy as np


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
