import pytest
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_allclose

from spicy_snow.processing.snow_index import (
    calc_delta_vv,
    calc_delta_cross_ratio,
    calc_delta_gamma,
    clip_delta_gamma_outlier,
    find_repeat_interval,
    calc_prev_snow_index,
    calc_snow_index,
    calc_snow_index_to_snow_depth,
)

# Minimal fixture for testing
@pytest.fixture
def sample_dataset():
    times = pd.date_range("2025-01-01", periods=3)
    data = xr.Dataset(
        {
            "vv": (("time", "y", "x"), np.array([
                [[1, 2],[3,4]],
                [[2,3],[4,5]],
                [[3,4],[5,6]]
            ], dtype=float)),
            "vh": (("time", "y", "x"), np.array([
                [[0.5, 1],[1.5,2]],
                [[1,1.5],[2,2.5]],
                [[1.5,2],[2.5,3]]
            ], dtype=float)),
            "fcf": (("y","x"), np.array([[0.2,0.5],[0.8,0.1]])),
            "track": ("time", np.array([1,1,1]))
        },
        coords={"time": times, "y":[0,1], "x":[0,1]},
        attrs={"s1_units":"dB"}
    )
    return data

def test_calc_delta_vv(sample_dataset):
    ds = calc_delta_vv(sample_dataset)
    # deltavv should have NaN for first time step
    np.testing.assert_array_equal(np.isnan(ds['deltavv'].isel(time=0)), np.ones((2,2), dtype=bool))
    # second time step should equal difference
    np.testing.assert_array_equal(ds['deltavv'].isel(time=1), np.array([[1,1],[1,1]]))

def test_calc_delta_cross_ratio(sample_dataset):
    ds = calc_delta_cross_ratio(sample_dataset, A=2)
    np.testing.assert_array_equal(np.isnan(ds['deltaCR'].isel(time=0)), np.ones((2,2), dtype=bool))
    # check that deltaCR second timestep is numeric
    assert not np.any(np.isnan(ds['deltaCR'].isel(time=1)))

def test_calc_delta_gamma(sample_dataset):
    ds = calc_delta_vv(sample_dataset)
    ds = calc_delta_cross_ratio(ds)
    ds = calc_delta_gamma(ds, B=0.5)
    assert 'deltaGamma' in ds
    # values should be combination of deltaCR and deltavv
    val = ds['deltaGamma'].isel(time=1).values
    assert np.all(val >= 0)

def test_clip_delta_gamma_outlier(sample_dataset):
    ds = calc_delta_vv(sample_dataset)
    ds = calc_delta_cross_ratio(ds)
    ds = calc_delta_gamma(ds)
    # artificially inflate values
    ds['deltaGamma'] = ds['deltaGamma'] + 100
    ds = clip_delta_gamma_outlier(ds, thresh=3)
    assert np.nanmax(ds['deltaGamma']) <= 3

def test_find_repeat_interval():
    # 6-day interval dataset
    times = pd.date_range("2025-01-01", periods=3, freq="6D")
    ds = xr.Dataset(
        {
            "vv": (("time", "y", "x"), np.random.rand(3, 2, 2)),
            "fcf": (("y", "x"), np.array([[0.2, 0.5], [0.8, 0.1]]))
        },
        coords={
            "time": times,
            "track": ("time", [1, 1, 1])
        },
        attrs={"s1_units": "dB"}
    )
    repeat = find_repeat_interval(ds)
    assert repeat.days == 6

def test_calc_prev_snow_index(sample_dataset):
    ds = sample_dataset.copy()
    ds['snow_index'] = xr.zeros_like(ds['vv'])
    repeat = pd.Timedelta('6D')
    prev = calc_prev_snow_index(ds, ds.time[2].values, repeat)
    assert prev.shape == (2,2)

def test_calc_snow_index_normal():
    """
    Normal test: snow_index accumulates deltaGamma correctly.
    First step of each track has a previous image (dummy zeros).
    """

    # 4 time steps: 1 dummy + 3 real
    times = pd.date_range("2025-01-01", periods=4, freq="6D")

    # deltaGamma: first step dummy zeros
    delta_gamma = np.zeros((4, 2, 2))
    delta_gamma[1:] = np.random.rand(3, 2, 2)

    snowcover = np.ones((4, 2, 2), dtype=bool)
    fcf = np.array([[0.2, 0.5], [0.8, 0.1]])

    # All steps same track
    tracks = [1, 1, 1, 1]

    ds = xr.Dataset(
        {
            "deltaGamma": (("time", "y", "x"), delta_gamma),
            "snowcover": (("time", "y", "x"), snowcover),
            "fcf": (("y", "x"), fcf)
        },
        coords={
            "time": times,
            "track": ("time", tracks)
        },
        attrs={"s1_units": "dB"}
    )

    # Run snow index calculation
    ds_out = calc_snow_index(ds)

    # Checks
    assert "snow_index" in ds_out.data_vars
    assert np.all(ds_out["snow_index"] >= 0)

    # First dummy step = 0
    np.testing.assert_allclose(ds_out['snow_index'].isel(time=0), delta_gamma[0])
    # Subsequent steps accumulate
    np.testing.assert_allclose(ds_out['snow_index'].isel(time=1), delta_gamma[0] + delta_gamma[1])
    np.testing.assert_allclose(ds_out['snow_index'].isel(time=2), delta_gamma[0] + delta_gamma[1] + delta_gamma[2])
    np.testing.assert_allclose(ds_out['snow_index'].isel(time=3), delta_gamma[0] + delta_gamma[1] + delta_gamma[2] + delta_gamma[3])

def test_calc_snow_index_no_previous_time():
    """
    Test calc_snow_index explicitly when there is no previous time step.
    The first time slice should use prev_si = 0, so snow_index == deltaGamma.
    """

    # 3 time steps, regular 6-day interval
    times = pd.date_range("2025-01-01", periods=3, freq="6D")
    ds = xr.Dataset(
        {
            "deltaGamma": (("time", "y", "x"), np.random.rand(3, 2, 2)),
            "snowcover": (("time", "y", "x"), np.ones((3, 2, 2), dtype=bool)),
        },
        coords={
            "time": times,
            "track": ("time", [1, 1, 1])
        },
        attrs={"s1_units": "dB"}
    )

    # Run snow index calculation
    ds_out = calc_snow_index(ds)

    # Check that 'snow_index' exists
    assert 'snow_index' in ds_out.data_vars

    # First time step has no previous images -> prev_si = 0
    np.testing.assert_array_equal(ds_out['snow_index'].isel(time=0), ds['deltaGamma'].isel(time=0))

    # Check that later time steps are >= 0 (snow_index accumulates deltaGamma)
    assert np.all(ds_out['snow_index'].isel(time=1) >= 0)
    assert np.all(ds_out['snow_index'].isel(time=2) >= 0)

def test_calc_snow_index_missing_first_time():
    """
    Test behavior when the first time step is missing or has NaNs.
    The function should propagate deltaGamma correctly for first valid time step.
    """

    times = pd.date_range("2025-01-01", periods=3, freq="6D")
    delta_gamma = np.random.rand(3, 2, 2)
    delta_gamma[0] = np.nan  # first time step missing

    ds = xr.Dataset(
        {
            "deltaGamma": (("time", "y", "x"), delta_gamma),
            "snowcover": (("time", "y", "x"), np.ones((3, 2, 2), dtype=bool)),
        },
        coords={
            "time": times,
            "track": ("time", [1, 1, 1])
        },
        attrs={"s1_units": "dB"}
    )

    ds_out = calc_snow_index(ds)

    # First snow_index slice should be NaN where deltaGamma is NaN
    np.testing.assert_array_equal(np.isnan(ds_out['snow_index'].isel(time=0)), np.isnan(delta_gamma[0]))

    # Later slices should be >= 0
    assert np.all(ds_out['snow_index'].isel(time=1) >= 0)
    assert np.all(ds_out['snow_index'].isel(time=2) >= 0)

def test_calc_snow_index_to_snow_depth(sample_dataset):
    ds = sample_dataset.copy()
    ds['snow_index'] = xr.ones_like(ds['vv'])
    ds = calc_snow_index_to_snow_depth(ds, C=0.5)
    np.testing.assert_array_equal(ds['snow_depth'], 0.5*np.ones((3,2,2)))


@pytest.fixture
def test_dataset():
    x = np.linspace(0, 9, 10)
    y = np.linspace(10, 19, 10)
    times = [np.datetime64(t) for t in ['2020-01-01T00:00','2020-01-01T00:10',
                                        '2020-01-02T10:10', '2020-01-02T10:20', '2020-01-02T10:40']]
    times_full = []
    [times_full.extend([t + pd.Timedelta(f'{i} days') for t in times]) for i in range(0, 5 * 12, 12)]

    track = np.tile(np.array([24, 24, 65, 65, 65]), reps=5)
    platforms = np.tile(np.array(['S1A', 'S1A', 'S1B', 'S1B', 'S1B']), reps=5)
    direction = np.tile(np.array(['descending', 'descending', 'ascending', 'ascending', 'ascending']), reps=5)

    vv = np.random.randn(10, 10, len(times_full))
    vh = np.random.randn(10, 10, len(times_full))

    ds = xr.Dataset(
        data_vars=dict(
            vv=(["x", "y", "time"], vv),
            vh=(["x", "y", "time"], vh),
        ),
        coords=dict(
            x=("x", x),
            y=("y", y),
            time=times_full,
            track=("time", track),
            platform=("time", platforms),
            flight_dir=("time", direction),
        )
    )
    ds.attrs['s1_units'] = 'dB'
    return ds

def test_delta_vv(test_dataset):
    orbit_ds = test_dataset.sel(time=test_dataset.track == 24)
    da1 = orbit_ds['vv'].isel(time=0)
    da2 = orbit_ds['vv'].isel(time=1)
    da3 = orbit_ds['vv'].isel(time=2)
    real2_1_diff = da2 - da1
    real3_2_diff = da3 - da2

    ds1 = calc_delta_vv(orbit_ds)
    assert_allclose(ds1['deltavv'].isel(time=1), real2_1_diff)
    assert_allclose(ds1['deltavv'].isel(time=2), real3_2_diff)

def test_delta_vv_errors(test_dataset):
    test_dataset.attrs['s1_units'] = 'amp'
    import pytest
    with pytest.raises(AssertionError):
        calc_delta_vv(test_dataset)

def test_delta_cross_ratio(test_dataset):
    orbit_ds = test_dataset.sel(time=test_dataset.track == 24)
    A = 2
    CR_ds = orbit_ds['vh'] * A - orbit_ds['vv']
    real2_1_diff = CR_ds.isel(time=1) - CR_ds.isel(time=0)
    real3_2_diff = CR_ds.isel(time=2) - CR_ds.isel(time=1)

    ds1 = calc_delta_cross_ratio(orbit_ds, A=A)
    assert_allclose(ds1['deltaCR'].isel(time=1), real2_1_diff)
    assert_allclose(ds1['deltaCR'].isel(time=2), real3_2_diff)

def test_delta_cr_errors(test_dataset):
    test_dataset.attrs['s1_units'] = 'amp'
    import pytest
    with pytest.raises(AssertionError):
        calc_delta_cross_ratio(test_dataset)
