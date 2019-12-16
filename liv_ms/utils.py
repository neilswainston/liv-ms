'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
import inspect


def to_str(fnc):
    '''Get string representation of a function or partial function.'''
    if not fnc:
        return 'None'

    try:
        return fnc.__name__
    except Exception as err:
        keywords = dict(fnc.keywords)

        for key, value in keywords.items():
            if inspect.isfunction(value):
                keywords[key] = value.__name__

        return '%s %s' % (fnc.func.__name__, keywords)
