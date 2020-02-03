'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-self-use
# pylint: disable=wrong-import-order
import unittest

from rdkit import Chem

from liv_ms import chem
import numpy as np


class TestChem(unittest.TestCase):
    '''Class to test methods in chem module.'''

    def test_get_fngrprnt_funcs(self):
        '''Test get_fngrprnt_funcs method of chem module.'''
        fngrprnt_funcs = chem.get_fngrprnt_funcs()
        self.assertEqual(len(fngrprnt_funcs), 20)

    def test_encode_desc(self):
        '''Test encode_desc method of chem module.'''
        encoded_desc = chem.encode_desc('CCO')
        self.assertEqual(len(encoded_desc), 200)

        for val in encoded_desc:
            self.assertIsInstance(val, float)

    def test_encode_fngrprnt(self):
        '''Test encode_fngrprnt method of chem module.'''

        for fngrprnt_func in chem.get_fngrprnt_funcs():
            encoded_fngrprnt = chem.encode_fngrprnt('CCO', fngrprnt_func)

            if not fngrprnt_func:
                self.assertEqual(encoded_fngrprnt, [])
            else:
                for val in encoded_fngrprnt:
                    self.assertIsInstance(val, (int, float))

    def test_get_similarities(self):
        '''Test get_similarities method of chem module.'''
        similarities = chem.get_similarities(
            ['CCO', 'CO'], Chem.RDKFingerprint)
        self.assertEqual(len(similarities), 3)

        for pair, val in similarities.items():
            self.assertIsInstance(val, float)

            if pair[0] == pair[1]:
                self.assertEqual(val, 0.0)
            else:
                self.assertNotEqual(val, 0.0)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
