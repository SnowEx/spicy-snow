import unittest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow.processing.snow_index import calc_delta_VV, calc_delta_cross_ratio,\
    calc_delta_gamma, clip_delta_gamma_outlier, find_repeat_interval, \
    calc_prev_snow_index, calc_snow_index, calc_snow_index_to_snow_depth

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

        assert_allclose(ds1['deltaVV'].isel(time = 1), real2_1_diff)
        
        assert_allclose(ds1['deltaVV'].isel(time = 2), real3_2_diff)
    
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

        assert_allclose(ds1['deltaCR'].isel(time = 1), real2_1_diff)
        
        assert_allclose(ds1['deltaCR'].isel(time = 2), real3_2_diff)
    
    def test_delta_cr_errors(self):
        """
        test that if units are amplitude calc_delta_vv raises AssertionErro
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)
        
        ds.attrs['s1_units'] = 'amp'

        self.assertRaises(AssertionError, calc_delta_cross_ratio, ds)
    
    def test_delta_gamma(self):
        backscatter = np.random.randn(10, 10, 4, 3)
        deltaVV = np.random.randn(10, 10 , 4)
        deltaCR = np.random.randn(10, 10 , 4)
        fcf = np.random.randn(10, 10 , 4)/100 + 0.5
        fcf[fcf < 0] = 0
        fcf[fcf > 1] = 1
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-08', '2020-01-09']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaVV = (["x", "y", "time"], deltaVV),
                fcf = (["x", "y", "time"], fcf),
                deltaCR = (["x", "y", "time"], deltaCR)
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 1, 24, 1])))

        ds = calc_delta_gamma(test_ds)
        deltaG_calc = ds['deltaGamma'].values.ravel()
        
        fcf = ds['fcf']
        cr = ds['deltaCR']
        vv = ds['deltaVV']
        B = 0.5
        deltaG_real = (1 - fcf) * cr + (fcf * B * vv)
        deltaG_real = deltaG_real.values.ravel()

        assert_allclose(deltaG_real[~np.isnan(deltaG_real)], deltaG_calc[~np.isnan(deltaG_calc)])
    
    def test_delta_gamma_clip(self):
        backscatter = np.random.randn(10, 10, 4, 3)
        deltaVV = np.random.randn(10, 10 , 4)
        deltaCR = np.random.randn(10, 10 , 4)
        fcf = np.random.randn(10, 10 , 4)/100 + 0.5
        fcf[fcf < 0] = 0
        fcf[fcf > 1] = 1
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-08', '2020-01-09']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaVV = (["x", "y", "time"], deltaVV),
                fcf = (["x", "y", "time"], fcf),
                deltaCR = (["x", "y", "time"], deltaCR)
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 1, 24, 1])))

        ds = calc_delta_gamma(test_ds)
        old = ds.copy(deep = True)
        ds = clip_delta_gamma_outlier(ds)

        # assert max and min have been clipped        
        self.assertLessEqual(ds['deltaGamma'].max(), 3, "Max should be 3 after clipping")
        self.assertGreaterEqual(ds['deltaGamma'].min(), -3, "Min should be -3 after clipping")

        # assert number of nans did not change
        self.assertEqual(ds['deltaGamma'].isnull().sum(), old['deltaGamma'].isnull().sum())

    def test_find_repeat_interval(self):
        backscatter = np.random.randn(10, 10, 3)
        times = [np.datetime64(t) for t in ['2020-01-01', '2020-01-07', '2020-01-13']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time"], backscatter)
            ),
            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                time = times,
                relative_orbit = (["time"], [24, 24, 24])))
        
        repeat = find_repeat_interval(test_ds)

        self.assertEqual(repeat.days, 6)

        self.assertEqual(type(repeat), pd.Timedelta)

        test_ds['time'] = [np.datetime64(t) for t in ['2020-01-01', '2020-01-13', '2020-01-25']]

        repeat = find_repeat_interval(test_ds)

        self.assertEqual(repeat.days, 12)

        test_ds['time'] = [np.datetime64(t) for t in ['2020-01-01', '2020-01-06', '2020-01-17']]

        self.assertRaises(AssertionError, find_repeat_interval, test_ds)

        # check with multiple orbits

        backscatter = np.random.randn(10, 10, 6, 3)
        deltaGamma = np.random.randn(10, 10 , 6)
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-07','2020-01-08', '2020-01-14', '2020-01-15']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma)
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 1, 24, 1, 24, 1])))
        
        repeat = find_repeat_interval(test_ds)

        self.assertEqual(repeat.days, 6)
    
    def test_previous_snow_index(self):
        backscatter = np.random.randn(10, 10, 3, 3)
        deltaGamma = np.random.randn(10, 10 , 3)
        times = [np.datetime64(t) for t in ['2020-01-06', '2020-01-07', '2020-01-13']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                snow_index = (["x", "y", "time"], np.zeros_like(deltaGamma))
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 24, 24])))
        
        # test if all previous snow indexes are zero
        prev_si = calc_prev_snow_index(test_ds, current_time = test_ds.isel(time = 2).time.values, repeat = pd.Timedelta('6 days'))
    
        assert_allclose(np.zeros_like(prev_si), prev_si)

        # test to see if weights are working
        test_ds['snow_index'].loc[dict(time = '2020-01-06')] = 1
        test_ds['snow_index'].loc[dict(time = '2020-01-07')] = 2

        prev_si = calc_prev_snow_index(test_ds, current_time = test_ds.isel(time = 2).time.values, repeat = pd.Timedelta('6 days'))

        # this should be (6 * 2 + 5 * 1) / (6 + 5) or 1.54545454 
        assert_allclose(prev_si, np.ones_like(prev_si)* 17/11)

        times = [np.datetime64(t) for t in ['2020-01-01', '2020-01-07', '2020-01-14']]
        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                snow_index = (["x", "y", "time"], np.zeros_like(deltaGamma))
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 24, 24])))
        
        test_ds['snow_index'].loc[dict(time = '2020-01-07')] = 2

        # test to see if all days outside +- 5 days are exclude
        prev_si = calc_prev_snow_index(test_ds, current_time = test_ds.isel(time = 2).time.values, repeat = pd.Timedelta('6 days'))

        # this should be (5 * 1 + 4 * 2) / (5 + 4) or 1.555555 
        assert_allclose(prev_si, np.ones_like(prev_si)* 2)

        # test for 12 day interval
        backscatter = np.random.randn(10, 10, 4, 3)
        deltaGamma = np.random.randn(10, 10 , 4)
        times = [np.datetime64(t) for t in ['2020-01-11', '2020-01-12', '2020-01-13', '2020-01-25']]
        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                snow_index = (["x", "y", "time"], np.zeros_like(deltaGamma))
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 24, 24, 24])))
        
        test_ds['snow_index'].loc[dict(time = '2020-01-11')] = 2
        test_ds['snow_index'].loc[dict(time = '2020-01-12')] = 5
        test_ds['snow_index'].loc[dict(time = '2020-01-13')] = 10

        # test to see if all days outside +- 5 days are exclude
        prev_si = calc_prev_snow_index(test_ds, current_time = test_ds.isel(time = 3).time.values, repeat = pd.Timedelta('12 days'))

        # this should be (10 * 12 + 5 * 11 + 2 * 10) / (10+ 11+ 12) or 1.555555 
        assert_allclose(prev_si, np.ones_like(prev_si)* 5.909090909090909)

        # check with multiple orbits

        backscatter = np.random.randn(10, 10, 6, 3)
        deltaGamma = np.random.randn(10, 10 , 6)
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-07','2020-01-08', '2020-01-14', '2020-01-15']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                snow_index = (["x", "y", "time"], np.zeros_like(deltaGamma)),
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 1, 24, 1, 24, 1])))
        
        test_ds['snow_index'].loc[dict(time = '2020-01-01')] = 2
        
        prev_si = calc_prev_snow_index(test_ds, current_time = test_ds.isel(time = 1).time.values, repeat = pd.Timedelta('6 days'))
        assert_allclose(prev_si, np.ones_like(prev_si)*2)

        prev_si = calc_prev_snow_index(test_ds, current_time = test_ds.isel(time = 2).time.values, repeat = pd.Timedelta('6 days'))
        assert_allclose(prev_si, np.ones_like(prev_si) * 6*2/(5+6))
    
    def test_snow_index(self):
        backscatter = np.random.randn(10, 10, 3, 3)
        deltaGamma = np.random.randn(10, 10 , 3)
        times = [np.datetime64(t) for t in ['2020-01-01', '2020-01-07', '2020-01-14']]
        ims = np.full((10, 10, 3), 4)
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                ims = (["x", "y", "time"], ims)
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 24, 24])))
        
        ds = calc_snow_index(test_ds)

        # first time slice should be all nans. There is no deltaGamma
        assert ds['snow_index'].isel(time = 0).sum() == 0
        # second time slice should be all delta gamma of that time slice (no other previous)
        np.allclose(ds['snow_index'].isel(time = 1), ds['deltaGamma'].isel(time = 1))
        # last time slice should just be deltaGamma @ t = 1 + deltaGamma @ t = 2
        np.allclose(ds['snow_index'].isel(time = 2), \
            ds['deltaGamma'].isel(time = 1) + ds['deltaGamma'].isel(time = 2))

        # check with multiple orbits

        backscatter = np.random.randn(10, 10, 6, 3)
        deltaGamma = np.random.randn(10, 10 , 6)
        ims = np.full((10, 10, 6), 4)
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-07','2020-01-08', '2020-01-14', '2020-01-15']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                ims = (["x", "y", "time"], ims),
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 1, 24, 1, 24, 1])))
        
        ds = calc_snow_index(test_ds)

        # first time slice should be all nans. There is no deltaGamma
        assert ds['snow_index'].isel(time = 0).sum() == 0

        first_delta_gamma = ds['deltaGamma'].isel(time = 1).where(ds['deltaGamma'].isel(time = 1) > 0, 0)
        assert_allclose(ds['snow_index'].isel(time = 1), first_delta_gamma)

        # should be snowindex at t==0 (0) * 6 + si @ t == 1 (which is deltaGamma @ t=1) * 5 / (6 + 5) + deltaGamma @ t = 2 
        second_delta_gamma = first_delta_gamma*5/(6+5) + ds['deltaGamma'].isel(time = 2)
        second_delta_gamma = second_delta_gamma.where(second_delta_gamma > 0, 0)
        assert_allclose(ds['snow_index'].isel(time = 2), second_delta_gamma)

        # check with ims = 2 @ some points

        backscatter = np.random.randn(10, 10, 6, 3)
        deltaGamma = np.random.randn(10, 10 , 6)
        ims = np.full((10, 10, 6), 4)
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-07','2020-01-08', '2020-01-14', '2020-01-15']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                deltaGamma = (["x", "y", "time"], deltaGamma),
                ims = (["x", "y", "time"], ims),
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 1, 24, 1, 24, 1])))
        
        test_ds['ims'].loc[dict(time = '2020-01-02', x = 5, y = 5)] = 2
        test_ds['deltaGamma'].loc[dict(time = '2020-01-01', x = 5, y = 5)] = 5
        test_ds['deltaGamma'].loc[dict(time = '2020-01-02', x = 5, y = 5)] = 5
        
        ds = calc_snow_index(test_ds)

        assert ds['snow_index'].sel(time = '2020-01-02', x = 5, y = 5) == 0

    def test_snow_index_to_depth(self):

        backscatter = np.random.randn(10, 10, 3, 3)
        snow_index = np.random.randn(10, 10 , 3)
        times = [np.datetime64(t) for t in ['2020-01-01', '2020-01-07', '2020-01-14']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
                snow_index = (["x", "y", "time"], snow_index)
            ),

            coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24, 24, 24])))
        
        test_ds['snow_index'].loc[dict(time = '2020-01-01')] = 1
        test_ds['snow_index'].loc[dict(time = '2020-01-07')] = 3

        ds = calc_snow_index_to_snow_depth(test_ds, C = 0.6)

        assert_allclose(ds['snow_depth'].isel(time = 0), test_ds['snow_index'].isel(time = 0)*0.6)

        assert_allclose(ds['snow_depth'].isel(time = 1), test_ds['snow_index'].isel(time = 1)*0.6)

