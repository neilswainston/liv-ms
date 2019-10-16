'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
import random
import sys

from liv_ms import similarity
from liv_ms.spectra import mona
import pandas as pd


def get_df(chem, spec):
    '''Get DataFrame.'''
    chem_df = pd.DataFrame(chem)
    spec_df = pd.DataFrame(spec)

    return chem_df.join(spec_df)


def _get_spec(row):
    '''Get spectrum as m/z, intensity pairs.'''
    return row[['m/z', 'I']]


def main(args):
    '''main method.'''
    chem, spec = mona.get_spectra(args[0], 5000)  # int(args[1]))
    df = get_df(chem, spec)

    spectra = df[['m/z', 'I']].values

    matcher = similarity.SpectraMatcher(spectra)

    for _ in range(16):
        query_idx = random.randint(0, len(spectra) - 1)
        print(df.loc[query_idx]['name'])
        res = matcher.search(spectra[query_idx])
        res_idx = list(zip(*[res[0], list(range(len(res[0])))]))

        for score, idx in sorted(res_idx, reverse=True)[:11]:
            if idx != query_idx:
                print(df.loc[idx]['name'], score)

        print()

    df.to_csv('mona.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
