'''
(c) University of Liverpool 2020

All rights reserved.

@author': 'neilswainston
'''
from datetime import date
import sys
import uuid


def to_massbank(data, out_filename):
    '''Convert to massbank file.'''
    with open(out_filename, 'w') as fle:
        for idx, entry in enumerate(data):
            entry['metadata']['ACCESSION'] = idx
            _write_metadata(entry['metadata'], fle)
            _write_peaks(entry['peaks'], fle)
            fle.write('//\n')


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
    fle.write('PK$SPLASH: %s' % str(uuid.uuid4()))
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
    data = [
        {
            'metadata': _get_metadata('water', 'H2O', 18.01056, 'O',
                                      'InChI=1S/H2O/h1H2'),
            'peaks': [12.6, 27128.2]
        }
    ]
    to_massbank(data, args[1])


if __name__ == '__main__':
    main(sys.argv[1:])
