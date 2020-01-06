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

_PART_PATTERN = r'(\d+(?:\.\d+)?)(?: )?(?:um|micron|microm)'


def encode_column(df):
    '''Encode column.'''
    dims = df['column'].apply(_get_dims)
    part_size = df['column'].apply(_get_part_size)


def _get_dims(row):
    '''Get dimensions.'''
    mtch = re.search(_DIM_PATTERN, row.lower())

    dims = [float('NaN'), float('NaN')]

    if mtch:
        dims = sorted(map(float, mtch.groups()))

    print(row, dims)

    return dims


def _get_part_size(row):
    '''Get particle size.'''
    mtch = re.search(_PART_PATTERN, row.lower())

    part_size = float('NaN')

    if mtch:
        part_size = float(mtch.group(1))

    print(row, part_size)

    return part_size


def main(args):
    '''main method.'''
    df = pd.read_csv(args[0])
    encode_column(df)


if __name__ == '__main__':
    main(sys.argv[1:])
