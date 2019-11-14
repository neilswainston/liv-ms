'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-member
import sys

from rdkit import Chem, DataStructs
from liv_ms import similarity


def get_similarities(smiles):
    '''Get similarities between chemicals represented by SMILES.'''
    sims = {}

    mols = [Chem.MolFromSmiles(sml) for sml in smiles]
    fps = [Chem.RDKFingerprint(mol) for mol in mols]

    for idx1, fp1 in enumerate(fps):
        for idx2 in range(idx1, len(fps)):
            sims[(smiles[idx1], smiles[idx2])] = \
                1 - DataStructs.FingerprintSimilarity(fp1, fps[idx2])

    return sims


def main(args):
    '''main method.'''
    print(get_similarities(args))


if __name__ == '__main__':
    main(sys.argv[1:])
