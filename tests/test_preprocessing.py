import pytest
import numpy as np
import pandas as pd
import xarray as xr

from spicy_snow.processing.s1_preprocessing import (
    amplitude_to_dB,
    dB_to_amplitude,
    s1_orbit_averaging,
    s1_clip_outliers,
    ims_water_mask,
    merge_s1_subsets,
    amplitude_to_dB
)
from spicy_snow.spicy_constants import s1_dual_pols


# -----------------------------
# Test amplitude <-> dB conversion
# -----------------------------
def test_amplitude_to_dB_and_back():
    times = pd.date_range("2025-01-01", periods=2)
    data = xr.Dataset(
        {pol: (("time","y","x"), np.ones((2,2,2))*10) for pol in s1_dual_pols},
        coords={"time": times, "y": [0,1], "x": [0,1]}
    )
    data.attrs['s1_units'] = 'amp'

    # Convert to dB
    db = amplitude_to_dB(data.copy())
    for pol in s1_dual_pols:
        assert np.allclose(db[pol], 10*np.log10(10))
    assert db.attrs['s1_units'] == 'dB'

    # Convert back to amplitude
    amp = dB_to_amplitude(db.copy())
    for pol in s1_dual_pols:
        np.testing.assert_allclose(amp[pol], 10)
    assert amp.attrs['s1_units'] == 'amp'


# -----------------------------
# Test orbit averaging
# -----------------------------
def test_s1_orbit_averaging():
    times = pd.date_range("2025-01-01", periods=4)
    track = [1,1,2,2]
    data = xr.Dataset(
        {pol: (("time","y","x"), np.ones((4,2,2))) for pol in s1_dual_pols},
        coords={"time": times, "y": [0,1], "x": [0,1]}
    )
    data = data.assign_coords(track=("time", track))
    data.attrs['s1_units'] = 'dB'

    out = s1_orbit_averaging(data)
    # Overall mean should remain 1
    for pol in s1_dual_pols:
        np.testing.assert_allclose(out[pol].mean().values, 1)


# -----------------------------
# Test clipping outliers
# -----------------------------
def test_s1_clip_outliers():
    times = pd.date_range("2025-01-01", periods=2)
    arr = np.array([[[0,10],[20,30]], [[-100,50],[200,0]]], dtype=float)
    data = xr.Dataset(
        {pol: (("time","y","x"), arr.copy()) for pol in s1_dual_pols},
        coords={"time": times, "y": [0,1], "x": [0,1]}
    )
    data.attrs['s1_units'] = 'dB'

    out = s1_clip_outliers(data)
    for pol in s1_dual_pols:
        # Outliers should be masked (nan)
        assert np.isnan(out[pol].values).any()


# -----------------------------
# Test water mask
# -----------------------------
def test_ims_water_mask():
    times = pd.date_range("2025-01-01", periods=2)
    data = xr.Dataset(
        {
            pol: (("time","y","x"), np.ones((2,2,2))) for pol in s1_dual_pols
        },
        coords={"time": times, "y": [0,1], "x": [0,1]}
    )
    # add watermask: mark (0,0) as water
    data['watermask'] = xr.DataArray(np.zeros((2,2,2)), dims=("time","y","x"))
    data['watermask'].loc[dict(y=0,x=0)] = 1

    out = ims_water_mask(data)
    for pol in s1_dual_pols:
        # water pixels should be masked
        assert np.isnan(out[pol].sel(y=0,x=0)).all()


# -----------------------------
# Test merging datasets
# -----------------------------
def test_merge_s1_subsets():
    times = pd.date_range("2025-01-01", periods=2)
    ds1 = xr.Dataset(
        {pol: (("time","y","x"), np.ones((2,2,2))) for pol in s1_dual_pols},
        coords={"time": times, "y":[0,1], "x":[0,1]}
    )
    ds2 = xr.Dataset(
        {pol: (("time","y","x"), np.zeros((2,2,2))) for pol in s1_dual_pols},
        coords={"time": times, "y":[0,1], "x":[0,1]}
    )
    merged = merge_s1_subsets({'ds1': ds1, 'ds2': ds2})
    for pol in s1_dual_pols:
        # merged dataset contains both datasets
        assert pol in merged