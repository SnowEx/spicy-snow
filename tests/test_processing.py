import unittest

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow.processing.s1_preprocessing import s1_amp_to_dB, s1_dB_to_amp, \
    merge_partial_s1_images

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

        s1_amp_to_dB(ds)

        dB = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()
        dB = dB[~np.isnan(dB)]

        assert np.allclose(dB, 10 * np.log10(amp))

    def test_amp_2_dB_zero_mask(self):
        """
        Test the masking of 0 values in amplitude to dB conversion
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        ds['s1'].isel(time = 0).sel(band = 'VV')[100, 100] = 0

        s1_amp_to_dB(ds)

        assert(np.isnan(ds['s1'].isel(time = 0).sel(band = 'VV')[100, 100]))
    
    def test_dB_2_amp(self):
        """
        Test the conversion of VV and VH from dB to amplitude
        """
        
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        dB = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()

        s1_dB_to_amp(ds)

        amp = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()

        assert(np.allclose(amp, dB))

    def test_same_orbit_merge_zeros(self):
        """
        Tests whether two images with close time stamps (< 10 min) will be combined
        to form 1 image from split images on arbitrary lines.
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        img_old = ds.isel(time = 0).sel(band = 'VV')['s1']
        img1 = img_old.copy(deep = True)
        img2 = img_old.copy(deep = True)
        img3 = img_old.copy(deep = True)
        future = ds.isel(time = 0)['time'] + pd.Timedelta('10 seconds')
        # cut arbitrary slices as 0
        img1[:10, :] = 0
        img2[10:50, :] = 0
        img3[50:, :] = 0
        # make 1 image more than 10 minutes away and add to Datarray
        img4 = ds.isel(time = 1).sel(band = 'VV')['s1']
        # make two future dates within 10 seconds of eachother
        future1 = ds.isel(time = 0)['time'] + pd.Timedelta('10 seconds')
        img2['time'] = future1
        future2 = future1 + pd.Timedelta('10 seconds')
        img3['time'] = future2
        # concatenate sliced images on time dimension
        ds = xr.concat([img1, img2, img3, img4], dim = 'time')
        ds = ds.to_dataset()
        # run merge and ensure merged image is created
        ds1 = merge_partial_s1_images(ds)
        # assert we only have 1 time stamp left (combined image)
        assert(len(ds1.time) == 2)
        # assert newly created time is 1st timestamp
        assert(ds1.isel(time = 0).time == ds.isel(time = 0).time)
        # assert new merged image is same as pre-sliced image
        assert(np.allclose(ds1.isel(time = 0)['s1'], img_old))
    
    def test_same_orbit_merge_nans(self):
        """
        Tests whether two images with close time stamps (< 10 min) will be combined
        to form 1 image from split images on arbitrary lines.
        """
        with open('./tests/test_data/2_img_ds', 'rb') as f:
            ds = pickle.load(f)

        img_old = ds.isel(time = 0).sel(band = 'VV')['s1']
        img1 = img_old.copy(deep = True)
        img2 = img_old.copy(deep = True)
        img3 = img_old.copy(deep = True)
        img4 = ds.isel(time = 1).sel(band = 'VV')['s1']
        # cut arbitrary slices as 0
        img1[:10, :] = np.nan
        img2[10:50, :] = np.nan
        img3[50:, :] = np.nan
        # make two future dates within 10 seconds of eachother
        future1 = ds.isel(time = 0)['time'] + pd.Timedelta('10 seconds')
        img2['time'] = future1
        future2 = future1 + pd.Timedelta('10 seconds')
        img3['time'] = future2
        # concatenate sliced images on time dimension
        ds = xr.concat([img1, img2, img3, img4], dim = 'time')
        ds = ds.to_dataset()
        # run merge and ensure merged image is created
        ds1 = merge_partial_s1_images(ds)
        # assert we only have 1 time stamp left (combined image)
        assert(len(ds1.time) == 2)
        # assert newly created time is 1st timestamp
        assert(ds1.isel(time = 0).time == ds.isel(time = 0).time)
        # assert new merged image is same as pre-sliced image
        assert(np.allclose(ds1.isel(time = 0)['s1'], img_old))
    
if __name__ == '__main__':
    unittest.main()