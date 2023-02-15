import unittest

import numpy as np
import xarray
import pickle
import pandas.testing as pd_testing

import sys
from os.path import expanduser
sys.path.append(expanduser('../'))
from spicy_snow.processing.s1_preprocessing import s1_amp_to_dB

class TestSentinel1PreProcessing(unittest.TestCase):
    """
    Test functionality of preprocessing functions for Sentinel-1
    """
    with open('../tests/test_data/2_img_ds', 'rb') as f:
        ds = pickle.load(f)
    
    def test_amp_2_dB(self, ds = ds):
        amp = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()
        amp = amp[amp != 0]

        s1_amp_to_dB(ds)

        dB = ds['s1'].isel(time = 0).sel(band = 'VV').values.ravel()
        dB = dB[np.isfinite(dB)]

        assert np.allclose(dB, 10 * np.log10(amp))
    
if __name__ == '__main__':
    unittest.main()