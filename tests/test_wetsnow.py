import unittest

import numpy as np
import pandas as pd
import xarray as xr
import pickle

import sys
from os.path import expanduser
sys.path.append(expanduser('.'))
from spicy_snow.processing.snow_index import calc_delta_gamma, clip_delta_gamma_outlier, calc_snow_index

from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_newly_frozen_snow,\
    id_wet_negative_si, flag_wet_snow

class TestWetSnowFlags(unittest.TestCase):
    """
    Test functionality of identify and setting wet snow paper from Lievens et al 2021
    """

    fcf = np.random.randn(10, 10)/10 + 0.5
    deltaVV = np.random.randn(10, 10, 6) * 3
    deltaCR = np.random.randn(10, 10, 6) * 3
    ims = np.full((10, 10, 6), 4, dtype = int)
    times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-07','2020-01-08', '2020-01-14', '2020-01-15']]
    x = np.linspace(0, 9, 10)
    y = np.linspace(10, 19, 10)
    lon, lat = np.meshgrid(x, y)

    test_ds = xr.Dataset(data_vars = dict(
                fcf = (["x", "y"], fcf),
                deltaVV = (["x", "y", "time"], deltaVV),
                deltaCR = (["x", "y", "time"], deltaCR),
                ims = (["x", "y", "time"], ims),
            ),
       coords = dict(
                lon = (["x", "y"], lon),
                lat = (["x", "y"], lat),
                band = ['VV', 'VH', 'inc'],
                time = times,
                relative_orbit = (["time"], [24,1,24,1,24,1])) 
)
    test_ds = calc_delta_gamma(test_ds)
    test_ds = clip_delta_gamma_outlier(test_ds)
    test_ds = calc_snow_index(test_ds)

    def test_id_newly_wet(self, ds = test_ds):
        """
        Test settings newly wet snow
        """

        ds['deltaCR'][0,0,0] = -2.1
        ds['fcf'][0,0] = 0.1

        ds['deltaVV'][0,1,0] = -2.1
        ds['fcf'][0,1] = 0.6

        ds['deltaCR'][0,2,0] = -1.9
        ds['fcf'][0,2] = 0.1

        ds['deltaVV'][0,3,0] = -1.9
        ds['fcf'][0,3] = 0.6

        ds = id_newly_wet_snow(ds)

        self.assertEqual(ds['wet_flag'][0,0,0], 1)
        self.assertEqual(ds['wet_flag'][0,1,0], 1)
        self.assertEqual(ds['wet_flag'][0,2,0], 0)
        self.assertEqual(ds['wet_flag'][0,3,0], 0)
    
    def test_newly_wet_assertion(self, ds = test_ds):

        self.assertRaises(AssertionError, id_newly_wet_snow , ds.drop(['fcf']))

    def test_id_refrozen(self, ds = test_ds):
        """
        Test settings re frozen snow
        """

        # delta gamma is > 2dB so should set freeze flag
        ds['deltaGamma'][0,0,0] = +2.1
        ds['fcf'][0,0] = 0.1

        # delta gamma is < 2dB so shouldn't set freeze flag
        ds['deltaGamma'][0,1,0] = +1.9
        ds['fcf'][0,1] = 0.1

        # delta gamma is > 2dB so should set freeze flag
        ds['deltaGamma'][0,2,0] = +2.1
        ds['fcf'][0,2] = 0.6

        # delta gamma is < 2dB so shouldn't set freeze flag
        ds['deltaGamma'][0,3,0] = +1.9
        ds['fcf'][0,3] = 0.6

        ds = id_newly_frozen_snow(ds)

        self.assertEqual(ds['freeze_flag'][0,0,0], 1)
        self.assertEqual(ds['freeze_flag'][0,1,0], 0)
        self.assertEqual(ds['freeze_flag'][0,2,0], 1)
        self.assertEqual(ds['freeze_flag'][0,3,0], 0)
    
    def test_newly_frozen_assertion(self, ds = test_ds):

        self.assertRaises(AssertionError, id_newly_frozen_snow , ds.drop(['deltaGamma']))
    
    def test_negative_snow_index_wet(self, ds = test_ds):

        # ims is not snow, and SI is negative so shouldn't flag
        ds['ims'][0,0,0] = 2
        ds['snow_index'][0,0,0] = -1

        # ims is snow and SI is negative so should flag
        ds['ims'][0,1,0] = 4
        ds['snow_index'][0,1,0] = -1

        # ims is not snow and si is positive so shouldn't flag
        ds['ims'][0,2,0] = 2
        ds['snow_index'][0,2,0] = 1

        # ims is snow but SI is positive so shouldn't flag
        ds['ims'][0,3,0] = 4
        ds['snow_index'][0,3,0] = 1

        # ims is snow and SI is negative so should flag
        ds['ims'][0,1,1] = 4
        ds['snow_index'][0,1,1] = -1


        # identify possible newly wet snow in regions with SI < 4 and IMS == 4
        ds = id_wet_negative_si(ds)

        assert ds['alt_wet_flag'][0,0,0] == 0
        assert ds['alt_wet_flag'][0,1,0] == 1
        assert ds['alt_wet_flag'][0,2,0] == 0
        assert ds['alt_wet_flag'][0,3,0] == 0
        assert ds['alt_wet_flag'][0,1,1] == 1

    def test_negative_snow_index_wet_threshold(self, ds = test_ds):
        """
        Testing negative SI wet snow threshold for non zero threshold
        """

        # ims is not snow, and SI is below threshold so shouldn't flag
        ds['ims'][0,0,0] = 2
        ds['snow_index'][0,0,0] = -2

        # ims is snow and SI is below threshold so should flag
        ds['ims'][0,1,0] = 4
        ds['snow_index'][0,1,0] = -2

        # ims is not snow and si is above threshold so shouldn't flag
        ds['ims'][0,2,0] = 2
        ds['snow_index'][0,2,0] = 1

        # ims is snow but SI is above threshold so shouldn't flag
        ds['ims'][0,3,0] = 4
        ds['snow_index'][0,3,0] = 1

        # ims is snow and SI is below threshold so should flag
        ds['ims'][0,1,1] = 4
        ds['snow_index'][0,1,1] = -2

        # ims is snow and SI is above threshold so should not flag
        ds['ims'][0,1,2] = 4
        ds['snow_index'][0,1,2] = -0.9


        # identify possible newly wet snow in regions with SI < 4 and IMS == 4
        ds = id_wet_negative_si(ds, wet_SI_thresh = -1)

        assert ds['alt_wet_flag'][0,0,0] == 0
        assert ds['alt_wet_flag'][0,1,0] == 1
        assert ds['alt_wet_flag'][0,2,0] == 0
        assert ds['alt_wet_flag'][0,3,0] == 0
        assert ds['alt_wet_flag'][0,1,1] == 1
        assert ds['alt_wet_flag'][0,1,2] == 0

    def test_id_wet_one_orbit(self):
        """
        Test id wet snow
        """
        fcf = np.random.randn(10, 10)/10 + 0.5
        deltaVV = np.random.randn(10, 10, 6) * 3
        deltaCR = np.random.randn(10, 10, 6) * 3
        ims = np.full((10, 10, 6), 4, dtype = int)
        s1 = np.random.randn(10, 10, 6, 3)

        # times = [np.datetime64(t) for t in ['2020-01-01','2020-01-02', '2020-01-07','2020-01-08', '2020-01-14', '2020-01-15']]
        # [24,1,24,1,24,1]
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-07', '2020-01-13','2020-01-19', '2020-01-25', '2020-01-31']]
        times = [np.datetime64(t) for t in ['2020-01-01','2020-01-13', '2020-01-25','2020-02-06', '2020-02-18', '2020-03-02']]
        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        import sys
        sys.path.append('/Users/zachkeskinen/Documents/spicy-snow/')
        from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_newly_frozen_snow, id_wet_negative_si, flag_wet_snow
        from spicy_snow.processing.snow_index import calc_delta_gamma, clip_delta_gamma_outlier, calc_snow_index

        ds = xr.Dataset(data_vars = dict(
                        fcf = (["x", "y"], fcf),
                        deltaVV = (["x", "y", "time"], deltaVV),
                        deltaCR = (["x", "y", "time"], deltaCR),
                        ims = (["x", "y", "time"], ims),
                        s1 = (["x", "y", "time", "band"], s1)
                    ),
            coords = dict(
                        lon = (["x", "y"], lon),
                        lat = (["x", "y"], lat),
                        band = ['VV', 'VH', 'inc'],
                        time = times,
                        relative_orbit = (["time"], [24,24,24,24,24,24]))
        ) 

        ds = calc_delta_gamma(ds)
        ds = clip_delta_gamma_outlier(ds)
        ds = calc_snow_index(ds)

        ds = id_newly_frozen_snow(ds)
        ds = id_wet_negative_si(ds)
        ds = id_newly_wet_snow(ds)

        # 0,0 is wet @ t0, dry @ t1, dry @ t 2, dry @ t 3
        xi , yi = 0, 0
        ds['wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 1

        ds['wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[3], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[3], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[3], x = xi, y = yi)] = 0

        # 1,1 is dry @ t0, wet @ t1, wet @ t 2
        xi , yi = 1, 1
        ds['wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0

        # 2,2 is wet @ t1, dry @ t1, wet @ t 2
        xi , yi = 2, 2
        ds['wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 1

        ds['wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0

        # 3,3 is nans all along
        xi , yi = 3, 3
        ds['wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[1], x = xi, y = yi)] = 1

        ds['wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0

        for t in ds.time:
            ds['s1'].loc[dict(time = t, x = xi, y = yi, band = 'VV')] = np.nan

        ds = flag_wet_snow(ds)

        # 0,0 is wet @ t0, dry @ t1, dry @ t 2
        xi, yi = 0, 0
        self.assertEqual(ds['wet_snow'].loc[dict(time = ds.time[0], x = xi, y = yi)], 1)
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[1], x = xi, y = yi)], 0)
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[2], x = xi, y = yi)], 0)

        # 1,1 is dry @ t0, wet @ t1, wet @ t 2
        xi , yi = 1, 1
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[0], x = xi, y = yi)], 0)
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[1], x = xi, y = yi)], 1)
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[2], x = xi, y = yi)], 1)

        # 2,2 is wet @ t1, dry @ t1, wet @ t 2
        xi , yi = 2, 2
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[0], x = xi, y = yi)], 1)
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[1], x = xi, y = yi)], 0)
        self.assertEqual( ds['wet_snow'].loc[dict(time = ds.time[2], x = xi, y = yi)], 1)

        # 3, 3 is nans all along
        xi , yi = 3, 3
        self.assertTrue(np.isnan(ds['wet_snow'].loc[dict(time = ds.time[0], x = xi, y = yi)].values))
        self.assertTrue(np.isnan(ds['wet_snow'].loc[dict(time = ds.time[1], x = xi, y = yi)].values))
        self.assertTrue(np.isnan(ds['wet_snow'].loc[dict(time = ds.time[2], x = xi, y = yi)].values))

    def test_id_wet_multiple_orbit(self):

        fcf = np.random.randn(10, 10)/10 + 0.5
        times = pd.date_range("2020-01-01", end = '2020-04-30', freq = '6D')
        n = len(times)
        deltaVV = np.random.randn(10, 10, n) + 0.1
        deltaCR = np.random.randn(10, 10, n) + 0.1
        ims = np.full((10, 10, n), 4, dtype = int)
        s1 = np.random.randn(10, 10, n, 3)


        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        ros = np.resize([1, 24], (n))

        ds = xr.Dataset(data_vars = dict(
                        fcf = (["x", "y"], fcf),
                        deltaVV = (["x", "y", "time"], deltaVV),
                        deltaCR = (["x", "y", "time"], deltaCR),
                        ims = (["x", "y", "time"], ims),
                        s1 = (["x", "y", "time", "band"], s1)
                    ),
            coords = dict(
                        lon = (["x", "y"], lon),
                        lat = (["x", "y"], lat),
                        band = ['VV', 'VH', 'inc'],
                        time = times,
                        relative_orbit = (["time"], ros)) 
        )

        ds = calc_delta_gamma(ds)
        ds = clip_delta_gamma_outlier(ds)
        ds = calc_snow_index(ds)

        ds = id_newly_frozen_snow(ds)
        ds = id_wet_negative_si(ds)
        ds = id_newly_wet_snow(ds)

        # Relative Orbit 1
        # 0,0 is wet @ t0, dry @ t2, dry @ t4, dry @ t6, wet @ t8, wet @ t 8, wet @ t10, dry @ t12, dry @ t14
        xi , yi = 0, 0
        ds['wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[0], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[2], x = xi, y = yi)] = 1

        ds['wet_flag'].loc[dict(time = ds.time[4], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[4], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[4], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[6], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[6], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[6], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[8], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[8], x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = ds.time[8], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[10], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[10], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[10], x = xi, y = yi)] = 0

        ds['wet_flag'].loc[dict(time = ds.time[12], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[12], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[12], x = xi, y = yi)] = 1

        ds['wet_flag'].loc[dict(time = ds.time[14], x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = ds.time[14], x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = ds.time[14], x = xi, y = yi)] = 0

        # relative orbit # 2
        # 1,1 is dry @ t1, dry @ t3, wet @ t5, wet @ t7, dry @ t9
        xi , yi = 1, 1
        t = ds.time[1]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        t = ds.time[3]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        t = ds.time[5]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        t = ds.time[7]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        t = ds.time[9]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 1

        # 2,2 is dry until t @ 10 then dry for odd orbits and perma-wet for evens
        xi , yi = 2, 2
        for ts in range(21):
            t = ds.time[ts]
            ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
            ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
            ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[10]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        t = ds.time[12]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        # 3,3 is dry until t @ 11 then dry for even and wet for odds until re freeze @ t = 17
        xi , yi = 3, 3
        for ts in range(21):
            t = ds.time[ts]
            ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
            ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
            ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[11]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[17]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 1

        ds = flag_wet_snow(ds)

        # Relative Orbit 1
        # 0,0 is wet @ t0, dry @ t2, dry @ t4, dry @ t6, wet @ t8, wet @ t 8, wet @ t10, dry @ t12, dry @ t14
        xi, yi = 0, 0
        assert ds['wet_snow'].loc[dict(time = ds.time[0], x = xi, y = yi)] == 1
        assert ds['wet_snow'].loc[dict(time = ds.time[2], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[4], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[6], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[8], x = xi, y = yi)] == 1
        assert ds['wet_snow'].loc[dict(time = ds.time[10], x = xi, y = yi)] == 1
        assert ds['wet_snow'].loc[dict(time = ds.time[12], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[14], x = xi, y = yi)] == 0

        # relative orbit # 2
        # 1,1 is dry @ t1, dry @ t3, wet @ t5, wet @ t7, dry @ t9
        xi , yi = 1, 1
        assert ds['wet_snow'].loc[dict(time = ds.time[1], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[3], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[5], x = xi, y = yi)] == 1
        assert ds['wet_snow'].loc[dict(time = ds.time[7], x = xi, y = yi)] == 1
        assert ds['wet_snow'].loc[dict(time = ds.time[9], x = xi, y = yi)] == 0


        # 2,2 is dry until t @ 10 then dry for odd orbits and perma-wet for evens
        xi , yi = 2, 2
        assert ds['wet_snow'].loc[dict(time = ds.time[0], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[1], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[2], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[9], x = xi, y = yi)] == 0
        assert ds['wet_snow'].loc[dict(time = ds.time[10], x = xi, y = yi)] == 1
        for t in range(10, 21, 2):
            assert ds['wet_snow'].loc[dict(time = ds.time[t], x = xi, y = yi)] == 1, f"{t}"
        for t in range(11, 21, 2):
            assert ds['wet_snow'].loc[dict(time = ds.time[t], x = xi, y = yi)] == 0, f"{t}"
        assert ds.time[11] > pd.to_datetime('2020-02-01')

        # 3,3 is dry until t @ 11 then dry for even and wet for odds until re freeze @ t = 17
        xi , yi = 3, 3
        for ts in range(21, 2):
            t = ds.time[ts]
            ds['wet_snow'].loc[dict(time = t, x = xi, y = yi)] = 0
        for ts in range(11, 17, 2):
            t = ds.time[ts]
            ds['wet_snow'].loc[dict(time = t, x = xi, y = yi)] = 1
        for ts in range(17, 21, 2):
            t = ds.time[ts]
            ds['wet_snow'].loc[dict(time = t, x = xi, y = yi)] = 0
    
    def test_perma_wet_seasons(self):

        fcf = np.random.randn(10, 10)/10 + 0.5
        times = pd.date_range("2020-01-01", end = '2020-12-31', freq = '6D')
        n = len(times)
        wet_flag = np.full((10, 10, n), 0.0)
        alt_wet_flag = np.full((10, 10, n), 0.0)
        freeze_flag = np.full((10, 10, n), 0.0)
        s1 = np.random.randn(10, 10, n, 3)
        ims = np.full((10, 10, n), 4, dtype = int)

        x = np.linspace(0, 9, 10)
        y = np.linspace(10, 19, 10)
        lon, lat = np.meshgrid(x, y)

        ros = np.resize([1, 24], (n))

        ds = xr.Dataset(data_vars = dict(
                        fcf = (["x", "y"], fcf),
                        wet_flag = (["x", "y", "time"], wet_flag),
                        alt_wet_flag = (["x", "y", "time"], alt_wet_flag),
                        freeze_flag = (["x", "y", "time"], freeze_flag),
                        s1 = (["x", "y", "time", "band"], s1),
                        ims = (["x", "y", "time"], ims)
                    ),
            coords = dict(
                        lon = (["x", "y"], lon),
                        lat = (["x", "y"], lat),
                        time = times,
                        band = ["VV", "VH", "inc"],
                        relative_orbit = (["time"], ros)) 
        )


        # 3,3 is wet @ t1, t3 then refreezes and is dry for odds t5 ->
        # then wet @ t11 , t13, and refreezes at t 15 ->
        # perma wet should not occur prior to t11 and then should be t11 ->
        # until august
        xi , yi = 3, 3

        t = ds.time[1]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[3]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[5]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 1

        t = ds.time[11]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[13]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 0

        t = ds.time[15]
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 1

        # set one to nan and be sure it is nan for perma wet
        t = ds.time[15]
        xi , yi = 4 ,4 
        ds['wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['alt_wet_flag'].loc[dict(time = t, x = xi, y = yi)] = 0
        ds['freeze_flag'].loc[dict(time = t, x = xi, y = yi)] = 1
        ds['s1'].loc[dict(time = t, x = xi, y = yi, band = 'VV')] = np.nan

        ds = flag_wet_snow(ds)

        # check no perma wet before february
        v = ds['perma_wet'].sel(time = slice('2020-01-01','2020-02-01')).loc[dict(x = 3, y = 3)] == 0
        self.assertTrue(v.all())

        # check perma wet after t 11 but before august
        v = ds['perma_wet'].sel(time = slice('2020-01-01','2020-02-01')).loc[dict(x = 3, y = 3)] == 0
        self.assertTrue(v.all())

        # check t 13 onwards is perma wet for orbit 24
        v = ds['perma_wet'].sel(time = ds.relative_orbit == 24).sel(time = slice('2020-02-01','2020-08-01')).loc[dict(x = 3, y = 3)] == 0.5
        v = v.sel(time = slice(ds.time[13], '2020-12-31'))
        self.assertTrue(v.all())

        # check that no perma wet after august
        v = ds['perma_wet'].sel(time = slice('2020-08-01','2020-12-31')).loc[dict(x = 3, y = 3)] == 0
        self.assertTrue(v.all())

        ## check wet set from perma wet
        # check t 11 onwards is wet for orbit 24
        v = ds['wet_snow'].sel(time = ds.relative_orbit == 24).sel(time = slice('2020-02-01','2020-08-01')).loc[dict(x = 3, y = 3)] == 1
        v = v.sel(time = slice(ds.time[11], '2020-12-31'))
        self.assertTrue(v.all())

        ## check wet set from perma wet
        # check no wet snow after august
        v = ds['wet_snow'].sel(time = slice('2020-08-01','2020-12-31')).loc[dict(x = 3, y = 3)] == 0
        self.assertTrue(v.all())

        ## check wet set before february
        self.assertEqual(ds['wet_snow'].sel(time = ds.time[1]).loc[dict(x = 3, y = 3)], 1)
        self.assertEqual(ds['wet_snow'].sel(time = ds.time[3]).loc[dict(x = 3, y = 3)], 1)
        self.assertEqual(ds['wet_snow'].sel(time = ds.time[5]).loc[dict(x = 3, y = 3)], 0)
        self.assertEqual(ds['wet_snow'].sel(time = ds.time[7]).loc[dict(x = 3, y = 3)], 0)

        self.assertTrue(np.isnan(ds['wet_snow'].loc[dict(time = ds.time[15], x = 4, y = 4)].values))
        
if __name__ == '__main__':
    unittest.main()