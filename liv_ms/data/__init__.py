'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
import numpy as np


def to_numpy(array_str, sep=','):
    '''Convert array_str to numpy.'''
    return np.fromstring(array_str[1:-1], sep=sep)
