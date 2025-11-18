import pytest
import numpy as np
import pandas as pd
import xarray as xr

from spicy_snow.utils.raster import da_to01, mosaic_group, combine_close_images

# ----------------------------
# da_to01 tests
# ----------------------------

def test_da_to01_basic():
    da = xr.DataArray([0, 50, 100])
    norm = da_to01(da, old_min=0, old_max=100)
    np.testing.assert_allclose(norm.values, [0.0, 0.5, 1.0])

def test_da_to01_mask_out_of_bounds():
    da = xr.DataArray([-10, 50, 200])
    norm = da_to01(da, old_min=0, old_max=100)
    assert np.isnan(norm.values[0])
    assert np.isnan(norm.values[2])
    assert norm.values[1] == 0.5

def test_da_to01_equal_min_max():
    da = xr.DataArray([0, 1, 2])
    with pytest.raises(ValueError):
        da_to01(da, old_min=1, old_max=1)

# ----------------------------
# mosaic_group tests
# ----------------------------

def test_mosaic_group_basic():
    times = pd.date_range("2025-01-01", periods=3)
    data = xr.DataArray(
        [[[1, np.nan], [np.nan, 4]],
         [[np.nan, 2], [3, np.nan]],
         [[0, 0], [0, 0]]],
        dims=['time','x','y'],
        coords={'time': times}
    )
    mosaic = mosaic_group(data)
    assert mosaic.sizes['x'] == 2
    assert mosaic.sizes['y'] == 2

    # Add time axis to expected
    expected = np.array([[[1, 2], [3, 4]]])
    np.testing.assert_array_equal(mosaic.values, expected)

def test_mosaic_group_time_coord():
    times = pd.date_range("2025-01-01", periods=2)
    data = xr.DataArray(
        np.zeros((2,2,2)),
        dims=['time','x','y'],
        coords={'time': times}
    )
    mosaic = mosaic_group(data)
    # Time coordinate should be the mean of input times
    assert mosaic['time'].values == times.mean().to_numpy()

# ----------------------------
# combine_close_images tests
# ----------------------------

def test_combine_close_images_groups():

    times = pd.to_datetime([
        "2025-01-01T00:00:00",
        "2025-01-01T00:01:00",
        "2025-01-01T00:05:00"
    ])
    
    # Make a 3D DataArray with y=2, x=2
    data = xr.DataArray(
        [
            [[1, np.nan], [np.nan, 4]],  # first time
            [[np.nan, 2], [3, np.nan]],  # second time
            [[0, 0], [0, 0]]             # third time
        ],
        dims=['time', 'y', 'x'],
        coords={'time': times}
    )

    grouped = combine_close_images(data, time_tol=pd.Timedelta('2min'))

    # Should produce 2 groups: first two combined, last alone
    assert len(grouped['time']) == 2

    # Check first group (mosaic of first two elements)
    expected_first = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(grouped.isel(time=0).values, expected_first)

    # Check second group (third element alone)
    expected_second = np.array([[0, 0], [0, 0]])
    np.testing.assert_array_equal(grouped.isel(time=1).values, expected_second)


def test_combine_close_images_with_exact_tolerance():

    times = pd.to_datetime([
        "2025-01-01T00:00:00",
        "2025-01-01T00:02:00"
    ])
    
    # Make a 3D DataArray with y=2, x=2
    data = xr.DataArray(
        [
            [[1, np.nan], [np.nan, 4]],  # first time
            [[np.nan, 2], [3, np.nan]]   # second time
        ],
        dims=['time', 'y', 'x'],
        coords={'time': times}
    )

    grouped = combine_close_images(data, time_tol=pd.Timedelta('2min'))

    # Edge case: exactly at tolerance → should be separate groups
    assert len(grouped['time']) == 2

    # Optional: check first mosaic
    expected_first = np.array([[1, np.nan], [np.nan, 4]])
    np.testing.assert_array_equal(grouped.isel(time=0).values, expected_first)

    # Optional: check second mosaic
    expected_second = np.array([[np.nan, 2], [3, np.nan]])
    np.testing.assert_array_equal(grouped.isel(time=1).values, expected_second)
