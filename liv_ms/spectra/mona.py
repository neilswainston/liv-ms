'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
import ijson


_NAME_MAP = {'kegg': 'kegg.compound',
             'molecular formula': 'formula',
             'total exact mass': 'monoisotopic_mass:float'}


def get_spectra(filename, num_spectra=float('inf')):
    '''Get spectra and metabolite ids.'''

    chemicals = []
    spectra = []

    for prefix, typ, value in ijson.parse(open(filename)):
        if prefix == 'item' and typ == 'start_map':
            chemical = {}
            spectrum = {}
        elif prefix == 'item.compound.item.inchi':
            chemical['inchi'] = value
        elif prefix == 'item.compound.item.names.item.name':
            if 'name' not in chemical:
                chemical['name'] = value
            # chemical['names'].append(value)
        elif prefix in ['item.compound.item.metaData.item.name',
                        'item.metaData.item.name']:
            name = _normalise_name(value.lower())
        elif prefix == 'item.compound.item.metaData.item.value':
            _parse_compound_metadata(name, value, chemical)
            name = None
        elif prefix == 'item.id':
            spectrum['id'] = value
        elif prefix == 'item.metaData.item.value':
            spectrum[name] = value
            name = None
        elif prefix == 'item.spectrum':
            values = [float(val) for term in value.split()
                      for val in term.split(':')]
            spectrum['m/z'] = values[0::2]
            spectrum['I'] = values[1::2]
        # elif prefix == 'item.tags.item.text':
        #    spectrum['tags'].append(value)
        elif prefix == 'item' and typ == 'end_map':
            chemicals.append(chemical)
            spectra.append(spectrum)

            if len(spectra) == num_spectra:
                break

    return chemicals, spectra


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
