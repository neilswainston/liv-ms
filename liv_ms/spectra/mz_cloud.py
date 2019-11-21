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
    '''Search function.'''
    df = pd.read_csv(filename, header=None)
    df.columns = ['query_name']

    # Search query_name:
    df = df.apply(_search_query, axis=1)

    # Write csv file:
    df[['query_name', 'id', 'name', 'link', 'synonyms']].to_csv(
        filename + '.csv')


def _search_query(row, max_page_num=512):
    '''Search mzCloud for name.'''

    # Search and scrape:
    search_url = '%s/compound/Search?Query=%s&page=%i'

    for page_num in range(1, max_page_num):
        page = requests.get(search_url %
                            (_DOMAIN, row['query_name'], page_num))

        has_results, updated_row = _search_page(row, page)

        if not has_results:
            break

        if 'id' in updated_row:
            return updated_row

    return row


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

        synonyms = sorted(list(set(_get_span_text(span)
                                   for span in p.find_all('span'))))

        if row['query_name'].lower() == name.lower() or \
                row['query_name'].lower() in map(lambda x: x.lower(),
                                                 synonyms):
            row['name'] = name
            row['id'] = href.split('/')[-1]
            row['link'] = _DOMAIN + href
            row['synonyms'] = synonyms
            break

    return True, row


def _get_span_text(span):
    '''Get span text.'''
    for child_span in span.find_all('span'):
        return _get_span_text(child_span)

    return span.text.replace(';  ', '') if span.text else None


def main(args):
    '''main method.'''
    search(args[0])


if __name__ == '__main__':
    main(sys.argv[1:])
