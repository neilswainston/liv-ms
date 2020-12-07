'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import sys

from bs4 import BeautifulSoup
import requests

import pandas as pd


_DOMAIN = 'https://www.mzcloud.org'


def search(filename):
    '''Search filename.'''
    if filename.endswith('.txt'):
        _search_txt(filename)
    elif filename.endswith('.xlsx'):
        _search_xl(filename)


def _search_txt(filename):
    '''Search function.'''
    df = pd.read_csv(filename, header=None, sep='\n')
    df.columns = ['Name']

    # Search query_name:
    df = df.apply(_search_query, axis=1)

    # Write csv file:
    df[['Name', 'id', 'exact_match', 'mzcloud_name', 'link', 'synonyms']
       ].to_csv(filename + '.csv')


def _search_xl(filename):
    '''Search function.'''
    xl = pd.ExcelFile(filename)

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name)

        # Search query_name:
        df = df.apply(_search_query, axis=1)

        # Write csv file:
        df.to_csv(sheet_name + '.csv')


def _search_df():
    '''Search DataFrame.'''


def _search_query(row, max_page_num=512, max_attempts=128):
    '''Search mzCloud for name.'''
    if pd.isna(row['Name']):
        return row

    err = None

    for _ in range(max_attempts):
        # Search and scrape:
        search_url = '%s/compound/Search?Query=%s&page=%i'

        for page_num in range(1, max_page_num):
            try:
                url = search_url % (_DOMAIN, row['Name'], page_num)
                page = requests.get(url, timeout=5)
            except requests.exceptions.ReadTimeout as err:
                continue

            has_results, updated_row = _search_page(row, page)

            if not has_results:
                break

            if 'id' in updated_row:
                return updated_row

        return row

    raise err


def _search_page(row, page):
    '''Search page.'''
    soup = BeautifulSoup(page.text, 'html.parser')

    # Get name, id and link:
    results = soup.find_all('div', {'class': 'row srp-item'})

    if not results:
        return False, row

    for result in results:
        a = result.find('a', {'class': 'srp-trivial-name dont-break-out'})
        href = a.attrs['href']
        name = _get_span_text(a)

        # Get synonyms:
        p = result.find('p', {'class': 'subtle-color'})

        if p:
            synonyms = sorted(list(set(_get_span_text(span)
                                       for span in p.find_all('span'))))

            row['exact_match'] = row['Name'].lower() == name.lower() or \
                (row['Name'].lower() in map(lambda x: x.lower(), synonyms))
            row['mzcloud_name'] = name
            row['id'] = href.split('/')[-1]
            row['link'] = _DOMAIN + href
            row['synonyms'] = synonyms
            print(row)
            break

    return True, row


def _get_span_text(span):
    '''Get span text.'''
    for child_span in span.find_all('span'):
        return _get_span_text(child_span)

    return span.text.replace(';  ', '').strip() if span.text else None


def main(args):
    '''main method.'''
    search(args[0])


if __name__ == '__main__':
    main(sys.argv[1:])
