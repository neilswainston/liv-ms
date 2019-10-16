'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
import sys

import ijson


_NAME_MAP = {'kegg': 'kegg.compound',
             'molecular formula': 'formula',
             'total exact mass': 'monoisotopic_mass:float'}


def get_spectra(filename, num_records=float('inf')):
    '''Get spectra and metabolite ids.'''

    records = []

    for prefix, typ, value in ijson.parse(open(filename)):
        if prefix == 'item' and typ == 'start_map':
            record = {'chemical': {}, 'spectrum': {}}
        elif prefix == 'item.compound.item.inchi':
            record['chemical']['inchi'] = value
        elif prefix == 'item.compound.item.names.item.name':
            if 'name' not in record['chemical']:
                record['chemical']['name'] = value
            # record['chemical']['names'].append(value)
        elif prefix in ['item.compound.item.metaData.item.name',
                        'item.metaData.item.name']:
            name = _normalise_name(value.lower())
        elif prefix == 'item.compound.item.metaData.item.value':
            _parse_compound_metadata(name, value, record)
            name = None
        elif prefix == 'item.id':
            record['spectrum']['id'] = value
        elif prefix == 'item.metaData.item.value':
            record['spectrum'][name] = value
            name = None
        elif prefix == 'item.spectrum':
            values = [float(val) for term in value.split()
                      for val in term.split(':')]
            record['spectrum']['m/z'] = values[0::2]
            record['spectrum']['I'] = values[1::2]
        # elif prefix == 'item.tags.item.text':
        #    record['spectrum']['tags'].append(value)
        elif prefix == 'item' and typ == 'end_map':
            records.append(record)

            if len(records) == num_records:
                break

    return None, None


def _parse_compound_metadata(name, value, record):
    '''Parses compound metadata.'''
    if name == 'chebi' and isinstance(value, str):
        value = value.replace('CHEBI:', '').split()[0]

    record['chemical'][_normalise_name(name)] = value


def _normalise_name(name):
    '''Normalises name in name:value pairs.'''
    if name in _NAME_MAP:
        return _NAME_MAP[name]

    return name.replace(':', '_')


def main(args):
    '''main method.'''
    spectra, met_ids = get_spectra(args[0], int(args[1]))


if __name__ == '__main__':
    main(sys.argv[1:])
