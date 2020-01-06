'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import re
import sys

import pandas as pd

_DIM_PATTERN = \
    r'(\d+(?:\.\d+)?)(?: )?(?:mm)?(?: )?(?:x|by)(?: )?(\d+(?:\.\d+)?) ?mm'


def encode_column(df):
    '''Encode column.'''
    dims = df['column'].apply(_get_dims)


def _get_dims(row):
    '''Get dimensions.'''
    mtch = re.search(_DIM_PATTERN, row.lower())

    dims = [float('NaN'), float('NaN')]

    if mtch:
        dims = sorted(map(float, mtch.groups()))
    else:
        print(row)

    return dims


def main(args):
    '''main method.'''
    df = pd.read_csv(args[0])
    encode_column(df)


if __name__ == '__main__':
    main(sys.argv[1:])
