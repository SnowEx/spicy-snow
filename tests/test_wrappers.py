import unittest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd
import xarray as xr
import shapely

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow import retrieve_snow_depth

class TestWrappers(unittest.TestCase):
    """
    Test functionality of wrapper functions
    """
    
    def test_error_checking_wrapper(self):
        """
        Test the calculation of change in VV between time steps of same
        relative orbit
        """
        area = shapely.geometry.box(-110, 45, -109, 46)
        dates = ('2020-01-01', '2020-01-04')

        self.assertRaises(AssertionError, retrieve_snow_depth, [-110, 45, -109, 67], dates, '/tmp', 'job_name_test')

        self.assertRaises(AssertionError, retrieve_snow_depth, area, '2020-01-04', '/tmp', 'job_name_test')

        self.assertRaises(AssertionError, retrieve_snow_depth, area, ['a', 'b', 'c'], '/tmp', 'job_name_test')

        # test work dir
        self.assertRaises(AssertionError, retrieve_snow_depth, area, dates, 1230)

        # test debug

        self.assertRaises(AssertionError, retrieve_snow_depth, area, dates, '/tmp', 'job_name_test', 'job_name_test', 'debug this')

        # test params
        self.assertRaises(AssertionError, retrieve_snow_depth, area, dates, '/tmp', 'job_name_test', 'job_name_test', True, 'out.nc', params = [10, 1])

        self.assertRaises(AssertionError, retrieve_snow_depth, area, dates, '/tmp', 'job_name_test', 'job_name_test', True, 'out.nc', params = 10)