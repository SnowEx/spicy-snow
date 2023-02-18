import unittest

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow.processing.snow_index import calc_delta_VV, calc_delta_cross_ratio,\
    calc_delta_gamma, clip_delta_gamma_outlier

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

        assert np.allclose(ds1['deltaVV'].isel(time = 1), real2_1_diff)
        
        assert np.allclose(ds1['deltaVV'].isel(time = 2), real3_2_diff)
    
    def test_delta_vv_errors(self):
        """
        test that if units are amplitude calc_delta_vv raises AssertionErro
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)
        
        ds.attrs['s1_units'] = 'amp'

        self.assertRaises(AssertionError, calc_delta_VV, ds)
    
    def test_delta_cr(self):
        """
        Test the calculation of change in VV between time steps of same
        relative orbit
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        test_A = 2

        CR1, CR2, CR3 = ds['s1'].sel(band = 'VH') * test_A - ds['s1'].sel(band = 'VV')
        real2_1_diff = CR2 - CR1
        real3_2_diff = CR3 - CR2

        ds1 = calc_delta_cross_ratio(ds, A = test_A)

        assert np.allclose(ds1['deltaCR'].isel(time = 1), real2_1_diff), "Differences don't match t2 - t1"
        
        assert np.allclose(ds1['deltaCR'].isel(time = 2), real3_2_diff), "Differences don't match t3 - t2"
    
    def test_delta_cr_errors(self):
        """
        test that if units are amplitude calc_delta_vv raises AssertionErro
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)
        
        ds.attrs['s1_units'] = 'amp'

        self.assertRaises(AssertionError, calc_delta_cross_ratio, ds)
    
    def test_delta_gamma(self):
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        ds = calc_delta_cross_ratio(ds)
        ds = calc_delta_VV(ds)
        ds = calc_delta_gamma(ds)
        deltaG_calc = ds['deltaGamma'].values.ravel()
        
        fcf = ds['fcf']
        cr = ds['deltaCR']
        vv = ds['deltaVV']
        B = 0.5
        deltaG_real = (1 - fcf) * cr + (fcf * B * vv)
        deltaG_real = deltaG_real.values.ravel()

        assert np.allclose(deltaG_real[~np.isnan(deltaG_real)], deltaG_calc[~np.isnan(deltaG_calc)])
    
    def test_delta_gamma_clip(self):
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        ds = calc_delta_cross_ratio(ds)
        ds = calc_delta_VV(ds)
        ds = calc_delta_gamma(ds)
        old = ds.copy(deep = True)
        ds = clip_delta_gamma_outlier(ds)

        # assert max and min have been clipped        
        self.assertEqual(ds['deltaGamma'].max(), 3, "Max should be 3 after clipping")
        self.assertEqual(ds['deltaGamma'].min(), -3, "Min should be -3 after clipping")

        # assert number of nans did not change
        self.assertEqual(ds['deltaGamma'].isnull().sum(), old['deltaGamma'].isnull().sum())
