'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import ast
import re

from liv_ms.data import mona, rt
import pandas as pd


_FLOW_RATE_PATTERN = r'(\d+(?:\.\d+)?)\s?([um])[lL]\s?/\s?min' + \
    r'(?:\s+at\s+(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))? min)?'

_FLOW_GRAD_PATTERN_1 = r'(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?' + \
    r'(?:/(\d+(?:\.\d+)?))?(?:\s+at\s+(\d+(?:\.\d+)?)' + \
    r'(?:-(\d+(?:\.\d+)?))? min)?\.?$'

_FLOW_GRAD_PATTERN_2 = r'(\d+(?:\.\d+)?)(?:min)?:(\d+(?:\.\d+)?)%'

_FLOW_GRAD_PATTERN_3 = r'(\d+(?:\.\d+)?) % (\w) ?to ?(\d+(?:\.\d+)?)' + \
    r' % (\w)\/(\d+(?:\.\d+)?) min'

_FLOW_GRAD_PATTERN_4 = r'linear from (\d+(?:\.\d+)?)\w\/(\d+(?:\.\d+)?)\w' + \
    r' at (\d+(?:\.\d+)?) min to (\d+(?:\.\d+)?)\w\/(\d+(?:\.\d+)?)\w' + \
    r' at (\d+(?:\.\d+)?) min(?:, hold (\d+(?:\.\d+)?) min' + \
    r' at (\d+(?:\.\d+)?)\w\/(\d+(?:\.\d+)?)\w, reequilibration' + \
    r' (\d+(?:\.\d+)?)\w\/(\d+(?:\.\d+)?)\w \((\d+(?:\.\d+)?) min\))?'

_SOL_REGEXP = r'(?:(\d+(?:\.\d+)?)' + \
    r'\:?(\d+(?:\.\d+)?)?\:?(\d+(?:\.\d+)?)?)?' + \
    r' ?([a-z\s]+)(?:\:([a-z\s]+))?(?:\:([a-z\s]+))?' + \
    r' ?(?:(\d+(?:\.\d+)?)\:?(\d+(?:\.\d+)?)?\:?(\d+(?:\.\d+)?)?)?'

_DIM_PATTERN = \
    r'(\d+(?:\.\d+)?)(?: )?(?:mm)?(?: )?(?:x|by)(?: )?(\d+(?:\.\d+)?) ?mm'

_PART_PATTERN = r'(\d+(?:\.\d+)?)(?: )?(?:um|micron|microm)'

_HYDROPHOBIC_PATTERN = r'C\d+|BEH'


def get_rt_data(filename, num_spec=1e32, regen_stats=True):
    '''Get RT data.'''
    if regen_stats:
        # Get spectra:
        df = mona.get_spectra(filename, num_spec=num_spec)

        # Clean data:
        df = _clean_ms_level(df)
        df = _clean_rt(df)
        df = _clean_flow_rate(df)
        df = _clean_gradient(df)

        # Encode column:
        _encode_column(df)

        # Get stats:
        stats_df = rt.get_stats(df)

        # Save stats_df:
        _save_stats(stats_df)
    else:
        stats_df = pd.read_csv(
            'mona_stats.csv',
            converters={'column values': ast.literal_eval,
                        'flow rate values': ast.literal_eval,
                        'gradient values': ast.literal_eval})

    return stats_df


def _save_stats(stats_df):
    '''Save stats.'''
    # Convert values to list to enable saving:
    for col_name in ['column values',
                     'flow rate values',
                     'gradient values']:
        stats_df.loc[:, col_name] = \
            stats_df[col_name].apply(
            lambda x: x if isinstance(x, float) else list(x))

    stats_df.to_csv('mona_stats.csv', index=False)


def _clean_ms_level(df):
    '''Clean MS level.'''
    return df[df['ms level'] == 'MS2']


def _clean_rt(df):
    '''Clean retention time.'''
    df = df.dropna(subset=['retention time'])

    res = df['retention time'].apply(_clean_rt_row)
    df.loc[:, 'retention time'] = res
    df.loc[:, 'retention time'] = df['retention time'].astype('float32')

    return df.dropna(subset=['retention time'])


def _clean_rt_row(val):
    '''Clean single retention time value.'''
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
        return float('NaN')


def _clean_flow_rate(df):
    '''Clean flow rate.'''
    df.loc[:, 'flow rate values'] = \
        df['flow rate'].apply(_clean_flow_rate_row)

    return df


def _clean_flow_rate_row(val):
    '''Clean single flow rate value.'''
    terms = []

    try:
        terms.extend([(0.0, float(val)),
                      (2**16, float(val))])
    except ValueError:

        val = val.lower()

        for term in val.split(','):
            term = term.strip()
            term = term.replace('min-1', '/min')
            mtch = re.match(_FLOW_RATE_PATTERN, term)

            if mtch:
                grps = mtch.groups()
                factor = 1.0 if grps[1] == 'm' else 1000.0
                rate = float(grps[0]) / factor

                if grps[2]:
                    terms.extend([(float(grps[2]), rate)])
                else:
                    terms.extend([(0.0, rate),
                                  (2**16, rate)])

                if grps[3]:
                    terms.extend([(float(grps[3]), rate)])
            else:
                terms.extend([(0.0, float('NaN')),
                              (2**16, float('NaN'))])

    return rt.get_timecourse_vals(list(zip(*terms)))


def _clean_gradient(df):
    '''Clean gradient.'''
    _clean_solvents(df)

    df.loc[:, 'gradient values'] = \
        df.apply(_clean_gradient_row, axis=1)

    return df


def _clean_solvents(df):
    '''Clean solvent columns.'''
    for col in ['solvent', 'solvent a', 'solvent a)', 'solvent b']:
        # Lowercase:
        df[col] = df[col].str.lower()

        # Replace chemicals:
        for trgt, rplcmnt in [('h2o', 'water'),
                              ('acn', 'acetonitrile'),
                              ('ch3cn', 'acetonitrile'),
                              ('acetonitril ', 'acetonitrile '),
                              ('meoh', 'methanol'),
                              ('hcooh', 'formic acid'),
                              ('fa', 'formic acid')]:
            df[col] = [val.replace(trgt, rplcmnt)
                       if pd.notnull(val) else val
                       for val in df[col]]

        # Special cases:
        # Replace ' with 0.01% formic acid' type fields:
        df[col] = [re.sub(r'\s+with.*', '', str(val).lower())
                   for val in df[col]]

        # Volumes
        df[col] = [re.sub(r'\d+(?:\.\d+)?\s?mm\s?', '', str(val).lower())
                   for val in df[col]]

        # Formic acid / formate:
        df[col] = [re.sub(r'\d+.\d+%\s?form(ate|ic acid)(\s?(?:in\s?)|-)?',
                          '', val)
                   if pd.notnull(val) else None
                   for val in df[col]]

        # Bracketed terms:
        df[col] = [re.sub(r'\s*\([^)]*\)', '', val)
                   if pd.notnull(val) else None
                   for val in df[col]]

        # mpa/b:
        df[col] = [re.sub(r'mp\w:\s', '', val)
                   if pd.notnull(val) else None
                   for val in df[col]]

        # Separators:
        sep = '/' if col == 'solvent' else ':'

        df[col] = [re.sub(r'\s?(/|;|,)\s?', sep, val)
                   if pd.notnull(val) else None
                   for val in df[col]]

        # Trim:
        df[col] = df[col].str.strip()

        # Replace '' and 'nan':
        df[col].replace({'nan': None, '': None}, inplace=True)


def _clean_gradient_row(row):
    '''Clean gradient value.'''
    terms = []

    solv_a, solv_b = _get_solvents(row)

    flow_grad = row['flow gradient']

    if pd.isnull(flow_grad):
        return rt.get_timecourse_vals(
            [[0.0, 2**16], [float('NaN'), float('NaN')]])

    mtch = re.match(_FLOW_GRAD_PATTERN_4, flow_grad)

    if mtch:
        grps = mtch.groups()

        # linear from 98A/2B at 0 min:
        prop_a = float(grps[0]) / 100
        prop_b = float(grps[1]) / 100

        terms.append([float(grps[2]),
                      prop_a * solv_a + prop_b * solv_b])

        # to 2A/98B at 13 min:
        prop_a = float(grps[3]) / 100
        prop_b = float(grps[4]) / 100

        terms.append([float(grps[5]),
                      prop_a * solv_a + prop_b * solv_b])

        # hold 6 min at 2A/98B
        if grps[7]:
            prop_a = float(grps[7]) / 100
            prop_b = float(grps[8]) / 100

            terms.append([float(grps[5]) + float(grps[6]),
                          prop_a * solv_a + prop_b * solv_b])

            # reequilibration 98A/2B (5 min):
            prop_a = float(grps[9]) / 100
            prop_b = float(grps[10]) / 100

            terms.append([float(grps[5]) + float(grps[6]) +
                          float(grps[11]),
                          prop_a * solv_a + prop_b * solv_b])
    else:
        for term in flow_grad.split(','):
            term = term.strip()
            mtch = re.match(_FLOW_GRAD_PATTERN_1, term)

            if mtch:
                grps = mtch.groups()

                prop_a = float(grps[0]) / 100
                prop_b = float(grps[1]) / 100

                terms.append([float(grps[3]), prop_a *
                              solv_a + prop_b * solv_b])

                if grps[4]:
                    terms.append(
                        [float(grps[4]), prop_a * solv_a + prop_b * solv_b])

            else:
                mtch = re.match(_FLOW_GRAD_PATTERN_2, term)

                if mtch:
                    grps = mtch.groups()
                    prop_a = float(grps[1]) / 100
                    prop_b = 1 - prop_a

                    terms.append([float(grps[0]), prop_a *
                                  solv_a + prop_b * solv_b])
                else:
                    mtch = re.match(_FLOW_GRAD_PATTERN_3, term)

                    if mtch:
                        grps = mtch.groups()
                        prop_a = float(grps[0]) / 100
                        prop_b = 1 - prop_a

                        if grps[1] == 'B':
                            prop_a, prop_b = prop_b, prop_a

                        terms.append([0.0, prop_a * solv_a + prop_b * solv_b])

                        prop_a = float(grps[2]) / 100
                        prop_b = 1 - prop_a

                        if grps[3] == 'B':
                            prop_a, prop_b = prop_b, prop_a

                        terms.append([float(grps[4]),
                                      prop_a * solv_a + prop_b * solv_b])
                    else:
                        return rt.get_timecourse_vals(
                            [[0.0, 2**16], [float('NaN'), float('NaN')]])

    return rt.get_timecourse_vals(list(zip(*terms)))


def _get_solvents(row):
    '''Get solvents.'''

    solv_tokens = row['solvent'].split('/') if row['solvent'] \
        else (None, None)

    sol_a = row['solvent a'] if row['solvent a'] \
        else (row['solvent a)'] if row['solvent a)'] else solv_tokens[0])

    sol_b = row['solvent b'] if row['solvent b'] else solv_tokens[1]

    return _get_solv_aqua_ratio(sol_a), \
        _get_solv_aqua_ratio(sol_b)


def _get_solv_aqua_ratio(sol):
    '''Get solvent aqueous ratio.'''
    if not sol:
        # Assume aqueous:
        return 0.0

    solv_aqua = {'acetonitrile': 1.0,
                 'ammonium acetate': 0.0,
                 'formic acid': 0.0,
                 'c isopropanol': 1.0,
                 'methanol': 1.0,
                 'water': 0.0
                 }

    mtch = re.match(_SOL_REGEXP, sol)

    if mtch:
        grps = list(mtch.groups())

        # Special case: no numerical data:
        if not grps[0] and not grps[6]:
            return solv_aqua.get(grps[3], 0.0)

        for pos in [0, 1, 2, 6, 7, 8]:
            grps[pos] = float(grps[pos]) / 100.0 \
                if grps[pos] else 0.0

        val = grps[0] * solv_aqua.get(grps[3], 0.0) + \
            grps[1] * solv_aqua.get(grps[4], 0.0) + \
            grps[2] * solv_aqua.get(grps[5], 0.0) + \
            grps[6] * solv_aqua.get(grps[3], 0.0) + \
            grps[7] * solv_aqua.get(grps[4], 0.0) + \
            grps[8] * solv_aqua.get(grps[5], 0.0)

        return val

    # Assume aqueous:
    return 0.0


def _encode_column(df):
    '''Encode column.'''
    col_vals = df.apply(_get_column_values, axis=1)

    # Fill NaNs:
    col_vals_df = pd.DataFrame(item for item in col_vals)
    col_vals_df.fillna(col_vals_df.mean(), inplace=True)

    df['column values'] = pd.Series(col_vals_df.values.tolist())


def _get_column_values(row):
    '''Get column values.'''
    if pd.notna(row['column']):
        return _get_dims(row) + [_get_part_size(row), _get_hydrophobic(row)]

    return [float('NaN')] * 4


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
