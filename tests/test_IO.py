import unittest

import pandas as pd
from datetime import datetime

import sys
from os.path import expanduser
sys.path.append(expanduser('.'))
from spicy_snow.IO.user_dates import get_input_dates

class TestIoHelperFunctions(unittest.TestCase):
    """
    Test functionality of identify and setting wet snow paper from Lievens et al 2021
    """
    
    def test_date_helper(self):

        self.assertEqual(('2019-08-01', '2020-01-01'), get_input_dates('2020-01-01'))

        self.assertEqual(('2020-08-01', '2020-12-01'), get_input_dates('2020-12-01'))

        self.assertEqual(('2020-08-01', '2020-12-01'), get_input_dates('2020-12-01T00:00:0000'))

        self.assertEqual(('2020-08-01', '2020-12-01'), get_input_dates(pd.to_datetime('2020-12-01T00:00:0000')))

        self.assertEqual(('2020-08-01', '2020-12-01'), get_input_dates(datetime(2020, 12, 1)))

        self.assertEqual(('2020-10-01', '2020-12-01'), get_input_dates(datetime(2020, 12, 1), '2020-10-01'))

        self.assertEqual(('2018-01-01', '2020-01-01'), get_input_dates('2020-01-01T01:00:0001', '2018-01-01'))

    def test_date_assertions(self):

        self.assertRaises(AssertionError, get_input_dates , '2128-01-01')

        self.assertRaises(AssertionError, get_input_dates , '2015-01-01')

        self.assertRaises(AssertionError, get_input_dates , '2017-01-01', '2018-01-01')