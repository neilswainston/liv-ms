'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=wrong-import-order
from functools import partial
import sys

import ijson

import numpy as np
import pandas as pd


_NAME_MAP = {'kegg': 'kegg.compound',
             'molecular formula': 'formula',
             'total exact mass': 'monoisotopic_mass:float'}


def get_spectra(filename, num_spec=float('inf')):
    '''Get spec and metabolite ids.'''
    data = []

    for prefix, typ, value in ijson.parse(open(filename)):
        if prefix == 'item' and typ == 'start_map':
            entry = {}
        elif prefix == 'item.compound.item.inchi':
            entry['inchi'] = value
        elif prefix == 'item.compound.item.names.item.name':
            if 'name' not in entry:
                entry['name'] = value
            # chemical['names'].append(value)
        elif prefix in ['item.compound.item.metaData.item.name',
                        'item.metaData.item.name']:
            name = _normalise_name(value.lower())
        elif prefix == 'item.compound.item.metaData.item.value':
            _parse_compound_metadata(name, value, entry)
            name = None
        elif prefix == 'item.id':
            entry['id'] = value
        elif prefix == 'item.metaData.item.value':
            entry[name] = value
            name = None
        elif prefix == 'item.spectrum':
            values = [float(val) for term in value.split()
                      for val in term.split(':')]
            entry['m/z'] = np.array(values[0::2])
            entry['I'] = np.array(values[1::2])
        # elif prefix == 'item.tags.item.text':
        #    spectrum['tags'].append(value)
        elif prefix == 'item' and typ == 'end_map':
            data.append(entry)

            if len(data) == num_spec:
                break

    # Prevent np array truncations:
    np.set_printoptions(threshold=sys.maxsize)

    return pd.DataFrame(data)


def filter_spec(spectra_df, fltr=None, cols=None):
    '''Filter.'''
    fltr_df = spectra_df[cols] if cols else spectra_df

    if fltr:
        for val in fltr:
            fltr_df = fltr_df[fltr_df[val[0]] == val[1]]

    return fltr_df


def normalise_inten(spectra_df):
    '''Normalise intensities.'''
    spectra_df.loc[:, 'I'] = spectra_df['I'].apply(_normalise_inten)


def group(spectra_df, col='smiles'):
    '''Group.'''
    grouped = []
    columns = []

    for _, group_df in spectra_df.groupby([col]):
        m_z = [val for lst in group_df['m/z'] for val in lst]
        inten = [val for lst in group_df['I'] for val in lst]

        # Sort by intensity:
        m_z, inten = list(zip(*sorted(list(zip(m_z, inten)))))

        group_df.drop(['m/z', 'I'], axis=1, inplace=True)
        grouped.append(group_df.iloc[0, :].tolist() + [m_z, inten])
        columns = group_df.columns.tolist() + ['m/z', 'I']

    return pd.DataFrame(grouped, columns=columns).set_index('index')


def filter_inten(spectra_df, min_inten=0.1):
    '''Filter by intensity.'''
    _filter_inten_part = partial(_filter_inten, min_inten=min_inten)

    spectra_df[['m/z', 'I']] = \
        spectra_df[['m/z', 'I']].apply(_filter_inten_part, axis=1)


def _parse_compound_metadata(name, value, chemical):
    '''Parses compound metadata.'''
    if name == 'chebi' and isinstance(value, str):
        value = value.replace('CHEBI:', '').split()[0]

    chemical[_normalise_name(name)] = value


def _normalise_name(name):
    '''Normalises name in name:value pairs.'''
    if name in _NAME_MAP:
        return _NAME_MAP[name]

    return name.replace(':', '_')


def _normalise_inten(inten):
    '''Normalise intensity.'''
    return inten / inten.max()


def _filter_inten(cols, min_inten):
    '''Filter by intensity.'''
    return pd.Series([list(val) for val in zip(*[[m, i]
                                                 for m, i in zip(*cols)
                                                 if i > min_inten])])


def main(args):
    '''main method.'''
    filename = args[0]
    num_spec = int(args[1])

    # Get spectra:
    spec_df = get_spectra(filename, num_spec=num_spec)
    spec_df['index'] = spec_df.index

    # Filter:
    spec_df = filter_spec(spec_df, cols=['index',
                                         'name',
                                         'monoisotopic_mass_float',
                                         'formula',
                                         'smiles',
                                         'inchi',
                                         'm/z',
                                         'I'])

    # Normalise intensities:
    normalise_inten(spec_df)

    # Group (by SMILES):
    spec_df = group(spec_df)

    filter_inten(spec_df)

    spec_df.to_csv(args[2], index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
