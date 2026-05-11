import pytest
import numpy as np
import pandas as pd
import xarray as xr

from spicy_snow.processing.wet_snow import (
    id_newly_wet_snow,
    id_newly_frozen_snow,
    id_wet_negative_si,
    flag_wet_snow
)

from spicy_snow.processing.snow_index import calc_delta_gamma, clip_delta_gamma_outlier, calc_snow_index

@pytest.fixture
def sample_dataset():
    times = pd.date_range("2025-01-01", periods=3, freq="6D")
    ds = xr.Dataset(
        {
            "deltavv": (("time", "y", "x"), np.array([[[0, -3],[1, -5]],
                                                       [[-2, 0], [0, -1]],
                                                       [[-3, -4], [2, 1]]])),
            "deltaCR": (("time", "y", "x"), np.array([[[0, -1],[2, -3]],
                                                      [[-2, 1],[0, -1]],
                                                      [[-3, -2],[1, 0]]])),
            "deltaGamma": (("time", "y", "x"), np.random.rand(3,2,2)),
            "snow_index": (("time", "y", "x"), np.array([[[0.1, -0.5],[0.3, 0.2]],
                                                         [[0.2, 0.1],[0.0, -0.2]],
                                                         [[-0.1, 0.0],[0.4, 0.5]]])),
            "snowcover": (("time", "y", "x"), np.array([[[True, True],[False, True]],
                                                       [[True, True],[True, True]],
                                                       [[False, True],[True, True]]])),
            "fcf": (("y", "x"), np.array([[0.6, 0.3],[0.8, 0.2]])),
            "vv": (("time", "y", "x"), np.random.rand(3,2,2))
        },
        coords={
            "time": times,
            "track": ("time", [1,1,1])
        },
        attrs={"s1_units": "dB"}
    )
    return ds


def test_id_newly_wet_snow(sample_dataset):
    ds = id_newly_wet_snow(sample_dataset, wet_thresh=-2)
    assert "wet_flag" in ds.data_vars
    # check that wet_flag only has 0 or 1 values
    vals = ds['wet_flag'].values
    assert np.all(np.logical_or(vals==0, vals==1) | np.isnan(vals))


def test_id_newly_frozen_snow(sample_dataset):
    ds = id_newly_frozen_snow(sample_dataset, freeze_thresh=0.5)
    assert "freeze_flag" in ds.data_vars
    vals = ds['freeze_flag'].values
    assert np.all(np.logical_or(vals==0, vals==1) | np.isnan(vals))


def test_id_wet_negative_si(sample_dataset):
    ds = id_wet_negative_si(sample_dataset, wet_SI_thresh=0)
    assert "alt_wet_flag" in ds.data_vars
    vals = ds['alt_wet_flag'].values
    assert np.all(np.logical_or(vals==0, vals==1) | np.isnan(vals))


def test_flag_wet_snow(sample_dataset):
    ds = id_newly_wet_snow(sample_dataset, wet_thresh=-2)
    ds = id_wet_negative_si(ds, wet_SI_thresh=0)
    ds = id_newly_frozen_snow(ds, freeze_thresh=0.5)
    ds = flag_wet_snow(ds)
    assert "wet_snow" in ds.data_vars
    assert "perma_wet" in ds.data_vars
    vals = ds['wet_snow'].values
    perma_vals = ds['perma_wet'].values
    # check values are 0 or 1
    assert np.all(np.logical_or(vals==0, vals==1) | np.isnan(vals))
    assert np.all(np.logical_or(perma_vals<=1, np.isnan(perma_vals)))

@pytest.fixture
def test_ds():
    fcf = np.random.randn(10, 10)/10 + 0.5
    deltaVV = np.random.randn(10, 10, 6) * 3
    deltaCR = np.random.randn(10, 10, 6) * 3
    vv = np.random.randn(10, 10, 6, 2)  # VV and VH
    ims = np.full((10, 10, 6), 4, dtype=int)
    times = pd.to_datetime(['2020-01-01','2020-01-02','2020-01-07','2020-01-08','2020-01-14','2020-01-15'])
    x = np.linspace(0, 9, 10)
    y = np.linspace(10, 19, 10)
    lon, lat = np.meshgrid(x, y)

    ds = xr.Dataset(
        data_vars=dict(
            fcf=(["x", "y"], fcf),
            deltavv=(["x", "y", "time"], deltaVV),
            deltaCR=(["x", "y", "time"], deltaCR),
            snowcover=(["x", "y", "time"], ims),
            vv=(["x", "y", "time", "band"], vv)
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            band=['VV', 'VH'],
            time=times,
            track=(["time"], [24,1,24,1,24,1])
        )
    )
    ds = calc_delta_gamma(ds)
    ds = clip_delta_gamma_outlier(ds)
    ds = calc_snow_index(ds)
    return ds


def test_id_newly_wet(test_ds):
    ds = test_ds.copy()

    # Set up test values
    ds['deltaCR'][0, 0, 0] = -2.1
    ds['fcf'][0, 0] = 0.1
    ds['deltavv'][0, 1, 0] = -2.1
    ds['fcf'][0, 1] = 0.6
    ds['deltaCR'][0, 2, 0] = -1.9
    ds['fcf'][0, 2] = 0.1
    ds['deltavv'][0, 3, 0] = -1.9
    ds['fcf'][0, 3] = 0.6

    # Run the function
    ds = id_newly_wet_snow(ds)

    # Assertions
    # Explicitly select band 'VV' to avoid ambiguous array comparison
    assert ds['wet_flag'].sel(band='VV').isel(x=0, y=0, time=0) == 1
    assert ds['wet_flag'].sel(band='VV').isel(x=0, y=1, time=0) == 1
    assert ds['wet_flag'].sel(band='VV').isel(x=0, y=2, time=0) == 0
    assert ds['wet_flag'].sel(band='VV').isel(x=0, y=3, time=0) == 0

    # Optional: check both bands if relevant
    assert (ds['wet_flag'].isel(x=0, y=0, time=0) == 1).any()

def test_newly_wet_assertion(test_ds):
    with pytest.raises(AssertionError):
        id_newly_wet_snow(test_ds.drop_vars(['fcf']))


def test_id_refrozen(test_ds):
    ds = test_ds
    ds['deltaGamma'][0,0,0] = 2.1
    ds['fcf'][0,0] = 0.1
    ds['deltaGamma'][0,1,0] = 1.9
    ds['fcf'][0,1] = 0.1
    ds['deltaGamma'][0,2,0] = 2.1
    ds['fcf'][0,2] = 0.6
    ds['deltaGamma'][0,3,0] = 1.9
    ds['fcf'][0,3] = 0.6

    ds = id_newly_frozen_snow(ds)

    assert ds['freeze_flag'][0,0,0] == 1
    assert ds['freeze_flag'][0,1,0] == 0
    assert ds['freeze_flag'][0,2,0] == 1
    assert ds['freeze_flag'][0,3,0] == 0


def test_newly_frozen_assertion(test_ds):
    with pytest.raises(AssertionError):
        id_newly_frozen_snow(test_ds.drop_vars(['deltaGamma']))


def test_flag_wet_snow_snowcover_mask_applied_to_all_timesteps():
    """Regression test for the stale-`ts` bug (PR #79): the final snowcover
    masking line in flag_wet_snow used a loop variable from outside the
    loop scope, so it only zeroed wet_snow at the last timestep instead of
    every timestep. Construct a multi-timestep dataset where snowcover is
    False in different cells per timestep and assert that wet_snow == 0
    at every (time, y, x) where snowcover is False.
    """
    # 6 timesteps over 2 orbits, big enough to get past flag_wet_snow's
    # >=4 melt-season check
    times = pd.date_range("2025-02-01", periods=6, freq="6D")
    n_t = len(times)
    rng = np.random.default_rng(42)
    # snowcover varies per timestep: a different cell is "no snow" each step
    snowcover = np.ones((n_t, 3, 3), dtype=bool)
    for t in range(n_t):
        snowcover[t, t % 3, t % 3] = False
    ds = xr.Dataset(
        {
            "deltavv": (("time", "y", "x"), rng.normal(0, 2, (n_t, 3, 3))),
            "deltaCR": (("time", "y", "x"), rng.normal(0, 2, (n_t, 3, 3))),
            "deltaGamma": (("time", "y", "x"), rng.normal(0, 1, (n_t, 3, 3))),
            "snow_index": (("time", "y", "x"), rng.normal(0.2, 0.1, (n_t, 3, 3))),
            "snowcover": (("time", "y", "x"), snowcover),
            "fcf": (("y", "x"), np.full((3, 3), 0.5)),
            "vv": (("time", "y", "x"), rng.normal(-15, 1, (n_t, 3, 3))),
        },
        coords={
            "time": times,
            "track": ("time", [10, 65, 10, 65, 10, 65]),
        },
        attrs={"s1_units": "dB"},
    )
    ds = id_newly_wet_snow(ds, wet_thresh=-2)
    ds = id_wet_negative_si(ds, wet_SI_thresh=0)
    ds = id_newly_frozen_snow(ds, freeze_thresh=0.5)
    ds = flag_wet_snow(ds)

    # Every cell with snowcover=False must end up with wet_snow=0 (or NaN
    # from upstream masking, but never >0).
    no_snow_mask = ~ds['snowcover'].values
    wet_at_no_snow = ds['wet_snow'].values[no_snow_mask]
    nonzero_at_no_snow = wet_at_no_snow[~np.isnan(wet_at_no_snow)] != 0
    assert not nonzero_at_no_snow.any(), (
        "wet_snow should be 0 (or NaN) wherever snowcover is False, "
        "across ALL timesteps — not just the last."
    )


# def test_negative_snow_index_wet(test_ds):
#     ds = test_ds
#     ds['snowcover'][0,0,0] = 2
#     ds['snow_index'][0,0,0] = -1
#     ds['snowcover'][0,1,0] = 4
#     ds['snow_index'][0,1,0] = -1
#     ds['snowcover'][0,2,0] = 2
#     ds['snow_index'][0,2,0] = 1
#     ds['snowcover'][0,3,0] = 4
#     ds['snow_index'][0,3,0] = 1
#     ds['snowcover'][0,1,1] = 4
#     ds['snow_index'][0,1,1] = -1

#     ds = id_wet_negative_si(ds)

#     # Select by x, y, time explicitly
#     assert ds['alt_wet_flag'].isel(x=0, y=0, time=0).item() == 0
#     assert ds['alt_wet_flag'].isel(x=0, y=1, time=0).item() == 0
#     assert ds['alt_wet_flag'].isel(x=0, y=2, time=0).item() == 1
#     assert ds['alt_wet_flag'].isel(x=0, y=3, time=0).item() == 1
