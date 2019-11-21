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


def search(filename):
    '''Search function.'''
    df = pd.read_csv(filename, header=None)
    df.columns = ['query_name']

    # Search query_name:
    df = df.apply(_search_query, axis=1)

    # Write csv file:
    df.to_csv(filename + '.csv')


def _search_query(row):
    '''Search mzCloud for name.'''
    domain = 'https://www.mzcloud.org'
    search_url = '%s/compound/Search?Query=%s'

    # Search and scrape:
    page = requests.get(search_url % (domain, row['query_name']))
    soup = BeautifulSoup(page.text, 'html.parser')

    # Get name, id and link:
    a = soup.find('a', {'class': 'srp-trivial-name dont-break-out'})

    if a:
        href = a.attrs['href']
        row['name'] = a.find('span').text
        row['id'] = href.split('/')[-1]
        row['link'] = domain + href

    # Get synonyms:
    p = soup.find('p', {'class': 'subtle-color'})

    if p:
        row['synonyms'] = sorted(list(set(_get_span_text(span)
                                          for span in p.find_all('span'))))

    return row


def _get_span_text(span):
    '''Get span text.'''
    for child_span in span.find_all('span'):
        return _get_span_text(child_span)

    return span.text


def main(args):
    '''main method.'''
    search(args[0])


if __name__ == '__main__':
    main(sys.argv[1:])
