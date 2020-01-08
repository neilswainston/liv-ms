'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
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

    return pd.DataFrame(data)


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
