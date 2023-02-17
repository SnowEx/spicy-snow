import unittest

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('.'))
from spicy_snow.processing.wet_snow import id_new_wet_snow, id_newly_frozen_snow,\
    id_wet_snow

class TestWetSnowMask(unittest.TestCase):
    """
    Test functionality of identify and setting wet snow paper from Lievens et al 2021
    """

    with open('/Users/zachkeskinen/Documents/spicy-snow/tests/test_data/2_img_ds', 'rb') as f:
        ds = pickle.load(f)

    def test_id_newly_wet(self, ds = ds):
        """
        Test settings newly wet snow
        """

        self.assertEqual()
    
    def test_id_refrozen(self, ds = ds):
        """
        Test settings re frozen snow
        """

        self.assertEqual()

    def test_id_wet(self, ds = ds):
        """
        Test id wet snow
        """

        self.assertEqual()
    
if __name__ == '__main__':
    unittest.main()