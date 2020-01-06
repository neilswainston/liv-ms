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

_HYDROPHOBIC_PATTERN = r'C\d+|BEH'


def encode_column(df):
    '''Encode column.'''
    df['column_values'] = df.apply(_get_column_values, axis=1)


def _get_column_values(row):
    '''Get column values.'''
    return _get_dims(row) + [_get_part_size(row), _get_hydrophobic(row)]


def _get_dims(row):
    '''Get dimensions.'''
    mtch = re.search(_DIM_PATTERN, row['column'].lower())

    dims = [float('NaN'), float('NaN')]

    if mtch:
        dims = sorted(map(float, mtch.groups()))

    return dims


def _get_part_size(row):
    '''Get particle size.'''
    mtch = re.search(_PART_PATTERN, row['column'].lower())

    part_size = float('NaN')

    if mtch:
        part_size = float(mtch.group(1))

    return part_size


def _get_hydrophobic(row):
    '''Get hydrophobic.'''
    return int(bool(re.search(_HYDROPHOBIC_PATTERN, row['column'])))


def main(args):
    '''main method.'''
    df = pd.read_csv(args[0])
    encode_column(df)
    print(df)


if __name__ == '__main__':
    main(sys.argv[1:])
