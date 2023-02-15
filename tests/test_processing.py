import unittest

import numpy as np
import xarray
import pickle
import pandas.testing as pd_testing

import sys
from os.path import expanduser
sys.path.append(expanduser('./'))
from spicy_snow.processing.s1_preprocessing import s1_amp_to_dB, s1_dB_to_amp

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
    
if __name__ == '__main__':
    unittest.main()