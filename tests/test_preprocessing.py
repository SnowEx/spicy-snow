import unittest

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow.processing.s1_preprocessing import s1_power_to_dB, s1_dB_to_power, \
    merge_partial_s1_images, s1_clip_outliers, s1_orbit_averaging, subset_s1_images, \
    merge_s1_subsets, s1_incidence_angle_masking

class TestSentinel1PreProcessing(unittest.TestCase):
    """
    Test functionality of preprocessing functions for Sentinel-1
    """
    
    def test_amp_2_dB_function(self):
        """
        Test the conversion of VV and VH from amplitude to dB
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        amp = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()
        amp = amp[amp != 0]

        ds_dB = s1_power_to_dB(ds)

        dB = ds_dB['s1'].isel(time = 0).sel(band = 'VV').values.ravel()
        dB = dB[~np.isnan(dB)]

        assert np.allclose(dB, 10 * np.log10(amp))

    def test_amp_2_dB_zero_mask(self):
        """
        Test the masking of 0 values in amplitude to dB conversion
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        ds['s1'].isel(time = 0).sel(band = 'VV')[100, 100] = 0

        ds_dB = s1_power_to_dB(ds)

        assert(np.isnan(ds_dB['s1'].isel(time = 0).sel(band = 'VV')[100, 100]))
    
    def test_dB_2_amp(self):
        """
        Test the conversion of VV and VH from dB to amplitude
        """
        
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        amp_original = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()

        ds_dB = s1_power_to_dB(ds)
        dB = ds_dB['s1'].isel(time = 0).sel(band = 'VV').values.ravel()

        ds_amp = s1_dB_to_power(ds_dB)
        amp = ds_amp['s1'].isel(time = 0).sel(band = 'VV').values.ravel()

        assert(np.allclose(amp, amp_original))
    
    def test_s1_partial_image_merge(self):

        backscatter = np.random.randn(10, 10, 25, 3)

        # make some np nan cuts out of backscatter images
        backscatter[:3, :, 0, :] = np.nan
        backscatter[3:, :, 1, :] = np.nan

        backscatter[:3, :, 2, :] = np.nan
        backscatter[3:7, :, 3, :] = np.nan
        backscatter[7:, :, 4, :] = np.nan


        times = [np.datetime64(t) for t in ['2020-01-01T00:00','2020-01-01T00:10','2020-01-02T10:10', '2020-01-02T10:20', '2020-01-02T10:40']]
        times_full = []
        [times_full.extend([t + pd.Timedelta(f'{i} days') for t in times]) for i in range(0, 5 * 12, 12)]
        orbits = np.tile(np.array([24, 24, 65, 65, 65]), reps = 5)

        abs_orbits = [31191, 31191, 28888, 28888, 28888]
        abs_full = []
        [abs_full.extend([o + i for o in abs_orbits]) for i in range(0, 5 * 12, 12)]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
            ),

            coords = dict(
                x = (["x"], x),
                y = (["y"], y),
                band = ['VV', 'VH', 'inc'],
                time = times_full,
                relative_orbit = (["time"], orbits),
                absolute_orbit = (["time"], abs_full)))
        
        assert np.sum(np.isnan(test_ds.isel(time = 0).sel(band = 'VV')['s1'])) == 30
        assert np.sum(np.isnan(test_ds.isel(time = 1).sel(band = 'VV')['s1'])) == 70
        assert np.sum(np.isnan(test_ds.isel(time = 2).sel(band = 'VV')['s1'])) == 30

        assert test_ds.time.size == 25

        merged = merge_partial_s1_images(test_ds)

        for band in ['VV','VH','inc']:
            assert np.sum(np.isnan(merged['s1'].isel(time = 0).sel(band = band))) == 0
            assert np.sum(np.isnan(merged['s1'].isel(time = 1).sel(band = band))) == 0

            self.assertEqual(merged.time.size, 10)

    def test_uneven_s1_partial_image_merge(self):

        backscatter = np.random.randn(10, 10, 25, 3)

        # make some np nan cuts out of backscatter images
        for i in range(0, 25, 5):
            backscatter[:3, :, i + 0, :] = np.nan
            backscatter[3:, :, i + 1, :] = np.nan

            backscatter[3:, :, i + 2, :] = np.nan
            backscatter[:3, :, i + 3, :] = np.nan
            backscatter[7:, :, i + 3, :] = np.nan
            backscatter[:7, :, i + 4, :] = np.nan


        times = [np.datetime64(t) for t in ['2020-01-01T00:00','2020-01-01T00:10','2020-01-02T10:10', '2020-01-02T10:20', '2020-01-02T10:40']]
        times_full = []
        [times_full.extend([t + pd.Timedelta(f'{i} days') for t in times]) for i in range(0, 5 * 12, 12)]
        orbits = np.tile(np.array([24, 24, 65, 65, 65]), reps = 5)

        abs_orbits = [31191, 31191, 28888, 28888, 28888]
        abs_full = []
        [abs_full.extend([o + i for o in abs_orbits]) for i in range(0, 5 * 12, 12)]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        test_ds = xr.Dataset(
            data_vars = dict(
                s1 = (["x", "y", "time", "band"], backscatter),
            ),

            coords = dict(
                x = (["x"], x),
                y = (["y"], y),
                band = ['VV', 'VH', 'inc'],
                time = times_full,
                relative_orbit = (["time"], orbits),
                absolute_orbit = (["time"], abs_full)))

        test_ds = test_ds.drop_isel(time = 0)
        test_ds = test_ds.drop_isel(time = 3)

        merged = merge_partial_s1_images(test_ds)

        for band in ['VV','VH','inc']:
            self.assertEqual(np.sum(np.isnan(merged['s1'].isel(time = 0).sel(band = band))), 70)
            self.assertEqual(np.sum(np.isnan(merged['s1'].isel(time = 1).sel(band = band))), 30)
            self.assertEqual(np.sum(np.isnan(merged['s1'].isel(time = 2).sel(band = band))), 0)
            self.assertEqual(np.sum(np.isnan(merged['s1'].isel(time = 3).sel(band = band))), 0)

            self.assertEqual(merged.time.size, 10)

    def test_outlier_clip(self):
        """
        Tests whether outliers 10th percentile - 3dB and 90th percentile + 3db
        are masked
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        t1 = ds.isel(time = 0)['time']
        for band in ['VV', 'VH']:
            ds.loc[dict(time = t1, band = band)]['s1'][0, 0] = -50
            ds.loc[dict(time = t1, band = band)]['s1'][0, 1] = 50

        ds1 = s1_clip_outliers(ds)

        assert ds['s1'].sel(band = 'VV').max() > ds1['s1'].sel(band = 'VV').max()
        assert ds['s1'].sel(band = 'VV').min() < ds1['s1'].sel(band = 'VV').min()
        assert ds['s1'].sel(band = 'VH').max() > ds1['s1'].sel(band = 'VV').max()
        assert ds['s1'].sel(band = 'VH').min() < ds1['s1'].sel(band = 'VV').min()
        
    def test_orbit_averaging(self):
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        ds = ds.assign_coords(relative_orbit = ('time', [0, 1, 2]))

        overall_means = ds['s1'].mean(dim = ['x', 'y', 'time']).sel(band = ['VV', 'VH'])

        aveds = s1_orbit_averaging(ds)

        ave_means = aveds.mean(dim = ['x', 'y']).sel(band = ['VV', 'VH'])

        for i in range(3):
            assert np.allclose(ave_means['s1'].sel(time = ave_means.relative_orbit == i), overall_means)

    def test_orbit_averaging_errors(self):
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)
        
        ds.attrs['s1_units'] = 'amp'

        self.assertRaises(AssertionError, s1_orbit_averaging, ds)

    x = np.linspace(0, 9, 10)
    y = np.linspace(10, 19, 10)
    lon, lat = np.meshgrid(x, y)
    times = pd.date_range("2020-01-01", end = '2020-12-31', freq = '6D')
    n = len(times)
    data = np.random.randn(10,10, n)

    ros = np.resize([1, 24], (n))
    platforms = np.resize(['S1A', 'S1B', 'S1B', 'S1A'], (n))
    direction = np.resize(['ascending', 'descending'], (n))

    test_ds = xr.Dataset(data_vars= dict(
        s1 = (["x", "y", "time"], data)
            ),
        coords = dict(
                            lon = (["x", "y"], lon),
                            lat = (["x", "y"], lat),
                            time = times,
                            relative_orbit = (["time"], ros),
                            platform = (["time"], platforms),
                            flight_dir = (["time"], direction)) 
            
        )

    def test_subset_dataset(self, ds = test_ds):
        
        dict_da = subset_s1_images(ds)
        
        v = dict_da['S1A-descending']['platform'] == 'S1A'
        self.assertTrue(v.all())

        v = dict_da['S1A-descending']['flight_dir'] == 'descending'
        self.assertTrue(v.all())

        v = dict_da['S1A-ascending']['platform'] == 'S1A'
        self.assertTrue(v.all())

        v = dict_da['S1A-ascending']['flight_dir'] == 'ascending'
        self.assertTrue(v.all())

        v = dict_da['S1B-ascending']['platform'] == 'S1B'
        self.assertTrue(v.all())

        v = dict_da['S1B-ascending']['flight_dir'] == 'ascending'
        self.assertTrue(v.all())

        v = dict_da['S1B-descending']['platform'] == 'S1B'
        self.assertTrue(v.all())

        v = dict_da['S1B-descending']['flight_dir'] == 'descending'
        self.assertTrue(v.all())
    
    def test_merge_subsets(self, ds = test_ds):

        dict_da = subset_s1_images(ds)

        merged_ds = merge_s1_subsets(dict_da)

        v = ds == merged_ds

        self.assertTrue(v.all())

    def test_incidence_angle_mask(self):

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


        test_ds['s1'].loc[dict(time = test_ds.time[0], x = 0, y = 0, band = 'inc')] = 2
        test_ds['s1'].loc[dict(time = test_ds.time[0], x = 0, y = 1, band = 'inc')] = 0.4
        ds = s1_incidence_angle_masking(test_ds)

        self.assertTrue(ds['s1'].sel(time = test_ds.time[0], x= 0, y = 0, band = 'VV').isnull())

        self.assertTrue(~ds['s1'].sel(time = test_ds.time[0], x= 0, y = 1, band = 'VV').isnull())

        
        
if __name__ == '__main__':
    unittest.main()