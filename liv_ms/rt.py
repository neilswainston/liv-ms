'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import sys

from liv_ms.spectra import mona


def _clean_ms_level(df):
    '''Clean MS level.'''
    return df[df['ms level'] == 'MS2']


def _clean_rt(df):
    '''Get spectra.'''
    df = df.dropna(subset=['retention time'])

    res = df['retention time'].apply(_clean_rt_row)
    df.loc[:, 'retention time'] = res
    df.loc[:, 'retention time'] = df['retention time'].astype('float32')

    return df


def _clean_rt_row(val):
    '''Clean single retention time value.'''
    val_orig = val

    try:
        val = val.replace('N/A', 'NaN')
        val = val.replace('min', '')

        if 's' in val:
            val = val.replace('sec', '')
            val = val.replace('s', '')
            return float(val) / 60.0
    except AttributeError:
        # Forgiveness, not permission. Assume float and pass:
        pass

    try:
        return float(val)
    except ValueError:
        print(val_orig)
        return float('NaN')


def main(args):
    '''main method.'''

    # Get spectra:
    df = mona.get_spectra(args[0])

    df = _clean_ms_level(df)
    df = _clean_rt(df)

    df.to_csv('rt.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
