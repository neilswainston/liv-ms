'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-member
import re

from rdkit import Chem


def parse(smiles):
    '''Parse SMILES.'''
    print(smiles)

    # Canonicalise:
    smiles = _canonicalise(smiles)

    print(smiles)

    # Get tokens:
    tokens = []
    _get_tokens(smiles, tokens)

    return tokens


def _canonicalise(smiles):
    '''Canonicalise.'''
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _get_tokens(substr, tokens):
    '''Get token.'''

    # Halogens (Br and Cl) or square bracketed terms:
    match = re.match(r'^(Br|Cl|\[[^]]*\])', substr)

    if match:
        group = match.group(1)
        tokens.append(group)
        _get_tokens(substr[len(group):], tokens)
        return

    # Number(s):
    match = re.match(r'^(%?\d+)', substr)

    if match:
        group = match.group(1)
        tokens[-1] += group
        _get_tokens(substr[len(group):], tokens)
        return

    if substr:
        tokens.append(substr[0])
        _get_tokens(substr[1:], tokens)


def main():
    '''main method.'''
    for smiles in ['C1=CC=CC=C1', 'C1:C:C:C:C:C1', 'c1ccccc1',
                   'O=Cc1ccc(O)c(OC)c1'
                   'CN=C=O',
                   'CN1CCC[C@H]1c2cccnc2',
                   'CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4',
                   'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1',
                   'CCc(c%99)ccc2[n+]%99ccc3c2[nH]c4c3cccc4',
                   'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1',
                   r'CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\CO']:

        tokens = parse(smiles)
        print(tokens)
        print()


if __name__ == '__main__':
    main()
