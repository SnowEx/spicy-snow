import unittest

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow.processing.snow_index import calc_delta_VV

class TestSnowIndex(unittest.TestCase):
    """
    Test functionality of snow_index calculation functions
    """
    
    def test_delta_vv(self):
        """
        Test the calculation of change in VV between time steps of same
        relative orbit
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        da1, da2, da3 = ds['s1'].sel(band = 'VV')
        real2_1_diff = da2 - da1
        real3_2_diff = da3 - da2

        ds1 = calc_delta_VV(ds)

        assert np.allclose(ds1['s1'].sel(band = 'deltaVV').isel(time = 1), real2_1_diff)
        
        assert np.allclose(ds1['s1'].sel(band = 'deltaVV').isel(time = 2), real3_2_diff)
    
    def test_delta_vv_errors(self):
        """
        test that if units are amplitude calc_delta_vv raises AssertionErro
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)
        
        ds.attrs['s1_units'] = 'amp'

        self.assertRaises(AssertionError, calc_delta_VV, ds)