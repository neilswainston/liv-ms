'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def one_hot_encode(values):
    '''One-hot encode values.'''
    label_encoder = LabelEncoder()
    return label_encoder, to_categorical(label_encoder.fit_transform(values))
