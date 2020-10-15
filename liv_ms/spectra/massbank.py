'''
(c) University of Liverpool 2020

All rights reserved.

@author': 'neilswainston
'''
# pylint: disable=invalid-name
from datetime import date
import sys
import uuid

import numpy as np
import pandas as pd


def convert(df, out_filename):
    '''Convert DataFrame to massbank.'''
    data = []

    df['METFRAG_MZ'] = df['METFRAG_MZ'].apply(_to_numpy)

    for _, row in df.iterrows():
        entry = {
            # name,monoisotopic_mass_float,smiles,m/z,I,METFRAG_MZ
            'metadata': _get_metadata(row['name'],
                                      row['formula'],
                                      row['monoisotopic_mass_float'],
                                      row['smiles'],
                                      row['inchi']),
            'peaks': row['METFRAG_MZ']
        }
        data.append(entry)
    to_massbank(data, out_filename)


def to_massbank(data, out_filename):
    '''Convert to massbank file.'''
    with open(out_filename, 'w') as fle:
        for idx, entry in enumerate(data):
            entry['metadata']['ACCESSION'] = idx
            _write_metadata(entry['metadata'], fle)
            _write_peaks(entry['peaks'], fle)
            fle.write('//\n')


def _to_numpy(array_str, sep=','):
    '''Convert array_str to numpy.'''
    return np.fromstring(array_str[1:-1], sep=sep)


def _write_metadata(metadata, fle):
    '''Write metadata.'''
    for key, value in metadata.items():
        if isinstance(value, list):
            for val in value:
                fle.write('%s: %s\n' % (key, val))
        else:
            fle.write('%s: %s\n' % (key, value))


def _write_peaks(peaks, fle):
    '''Write peaks.'''
    fle.write('PK$SPLASH: %s\n' % str(uuid.uuid4()))
    fle.write('PK$NUM_PEAK: %i\n' % len(peaks))
    fle.write('PK$PEAK: m/z int. rel.int.\n')

    for peak in peaks:
        fle.write('  %f 1 1\n' % peak)


def _get_metadata(name, formula, mass, smiles, inchi):
    '''Get example metadata.'''
    return {
        'RECORD_TITLE': name,
        'DATE': date.today().strftime('%Y.%m.%d'),
        'AUTHORS': 'Neil Swainston, University of Liverpool',
        'LICENSE': 'CC BY',
        'CH$NAME': name,
        'CH$COMPOUND_CLASS': 'N/A; Metabolomics Standard',
        'CH$FORMULA': formula,
        'CH$EXACT_MASS': mass,
        'CH$SMILES': smiles,
        'CH$IUPAC': inchi,
        'AC$INSTRUMENT': 'MetFrag',
        'AC$INSTRUMENT_TYPE': 'MetFrag',
        'AC$MASS_SPECTROMETRY': ['MS_TYPE MS2', 'ION_MODE POSITIVE'],
        'MS$FOCUSED_ION': 'BASE_PEAK %f' % mass
    }


def main(args):
    '''main method.'''
    convert(pd.read_csv(args[0]), args[1])


if __name__ == '__main__':
    main(sys.argv[1:])
