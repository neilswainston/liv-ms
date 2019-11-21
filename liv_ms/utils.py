'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
import inspect


def to_str(partial_func):
    '''Get string representation of a partial function.'''
    keywords = partial_func.keywords

    for key, value in keywords.items():
        if inspect.isfunction(value):
            keywords[key] = value.__name__

    return '%s %s' % (partial_func.func.__name__, keywords)
