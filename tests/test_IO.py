import unittest

import numpy as np
import pandas as pd
from datetime import datetime
import shapely.geometry
import xarray as xr

import sys
from os.path import expanduser
sys.path.append(expanduser('.'))
from spicy_snow.IO.user_dates import get_input_dates
from spicy_snow.IO.user_area import get_input_area

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
    
    def test_area_helper(self):

        area = shapely.geometry.box(-115, 41, -113, 43)

        self.assertEqual(get_input_area([-115, 41, -113, 43]), area)

        self.assertEqual(get_input_area([-115.0, 41.0, -113.0, 43.0]), area)

        self.assertEqual(get_input_area(['-115.0', '41.0', '-113.0', '43.0']), area)

        self.assertEqual(get_input_area([-115.0, 41.0, -113.0, 43.0]), area)

        area = shapely.geometry.box(-0.5, 9.5, 9.5, 19.5)

        x = np.arange(0, 10, 1)
        y = np.arange(10, 20, 1)

        ds = xr.Dataset(data_vars= dict(
            data = (["x", "y"], np.random.randn(10, 10))),
                coords = dict(
                        x = (["x"], x),
                        y = (["y"], y)),
        )
        ds = ds.rio.write_crs('EPSG:4326')

        self.assertEqual(get_input_area(img = ds), area)
    
    def test_area_assertions(self):

        self.assertRaises(AssertionError, get_input_area, [-181, 41, -113, 43])

        self.assertRaises(AssertionError, get_input_area, [-179, 41, 203, 43])

        self.assertRaises(AssertionError, get_input_area, [-179, 91, 179, 43])

        self.assertRaises(AssertionError, get_input_area, [-179, 43, 179, 91])

        self.assertRaises(AssertionError, get_input_area, [-179, -91, 179, 89])

        self.assertRaises(AssertionError, get_input_area, [-140, -89, -141, 89])

        self.assertRaises(AssertionError, get_input_area, [-142, 42, -141, 40])

        self.assertRaises(AssertionError, get_input_area, [-142, 42, -141, 40], 'fd')

        self.assertRaises(AssertionError, get_input_area)
