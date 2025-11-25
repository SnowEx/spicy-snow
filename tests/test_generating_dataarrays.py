import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from shapely.geometry import box
import tempfile

from spicy_snow.utils.raster import da_to01, combine_close_images
from spicy_snow.processing.generate_dataarrays import preallocate_output, convert_snowcover_dates_to_s1_overpasses, generate_forest_fraction_dataarray
from spicy_snow.utils.checks import validate_aoi, within_conus

# -----------------------------
# Test da_to01
# -----------------------------
def test_da_to01_basic():
    data = xr.DataArray([[0, 50], [100, 150]])
    norm = da_to01(data)
    expected = np.array([[0, 0.5], [1, np.nan]])
    np.testing.assert_array_equal(np.isnan(norm.values), np.isnan(expected))
    np.testing.assert_almost_equal(np.nan_to_num(norm.values), np.nan_to_num(expected))

def test_da_to01_error_on_equal_min_max():
    da = xr.DataArray([1,1,1])
    with pytest.raises(ValueError):
        da_to01(da, old_min=5, old_max=5)

# -----------------------------
# Test preallocate_output
# -----------------------------
def test_preallocate_output_creates_da():
    da_ref = xr.DataArray(np.zeros((2,2)), dims=("y","x"), coords={"y":[0,1],"x":[0,1]})
    da_ref = da_ref.rio.write_crs("EPSG:4326")
    times = pd.date_range("2025-01-01", periods=3)
    da_out = preallocate_output(times, da_ref.y, da_ref.x, da_ref.dtype, da_ref.rio.crs, da_ref.rio.transform())
    assert da_out.sizes["time"] == 3
    assert da_out.sizes["y"] == 2
    assert da_out.sizes["x"] == 2
    assert da_out.rio.crs.to_string() == "EPSG:4326"

# -----------------------------
# Test combine_close_images
# -----------------------------
def test_combine_close_images_3d():
    times = pd.date_range("2025-01-01", periods=3)
    da = xr.DataArray(
        [[[1, np.nan], [np.nan, 4]],
         [[np.nan, 2], [3, np.nan]],
         [[0, 0], [0, 0]]],
        dims=['time','y','x'],
        coords={'time': times}
    )
    grouped = combine_close_images(da, time_tol=pd.Timedelta("2min"))
    assert 'time' in grouped.dims
    assert grouped.sizes['y'] == 2
    assert grouped.sizes['x'] == 2

# -----------------------------
# Test convert_snowcover_dates_to_s1_overpasses
# -----------------------------
def test_convert_snowcover_dates_to_s1_overpasses_basic():
    times_s1 = pd.date_range("2025-01-01", periods=1)  # only 1 time
    s1_da = xr.DataArray(np.zeros((1,2,2)), dims=["time","y","x"], coords={"time":times_s1})
    viirs_times = pd.date_range("2025-01-01", periods=1)
    viirs = xr.DataArray(np.ones((1,2,2)), dims=["time","y","x"], coords={"time":viirs_times})
    out = convert_snowcover_dates_to_s1_overpasses(viirs, s1_da)
    np.testing.assert_array_equal(out.isel(time=0).values, np.ones((2,2)))

def test_convert_snowcover_dates_to_s1_overpasses_missing_date():
    times_s1 = pd.date_range("2025-01-01", periods=2)
    s1_da = xr.DataArray(np.zeros((2,2,2)), dims=["time","y","x"], coords={"time":times_s1})
    
    # Only one VIIRS date (missing second S1 date)
    viirs_times = pd.date_range("2025-01-01", periods=1)
    viirs = xr.DataArray(np.ones((1,2,2)), dims=["time","y","x"], coords={"time":viirs_times})

    with pytest.raises(ValueError, match="No VIIRS snowcover found for S1 date"):
        convert_snowcover_dates_to_s1_overpasses(viirs, s1_da)

# -----------------------------
# Test generate_forest_fraction_dataarray (mock download)
# -----------------------------
def test_generate_forest_fraction_dataarray_inside_conus(monkeypatch):
    aoi = box(-100, 30, -99, 31)
    
    # Patch download_proba_v to return temporary file
    def fake_download(*args, **kwargs):
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        da = xr.DataArray(np.array([[50,100],[0,25]]))
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.to_raster(tmp.name)
        return tmp.name
    
    monkeypatch.setattr("spicy_snow.utils.download.download_proba_v", fake_download)
    
    da = generate_forest_fraction_dataarray(aoi)
    assert da.max() <= 1
    assert da.min() >= 0

# -----------------------------
# Test validate_aoi and within_conus
# -----------------------------
def test_validate_aoi_box_list_and_dict():
    lst = [-125, 25, -66, 50]
    aoi1 = validate_aoi(lst)
    assert aoi1.bounds == tuple(lst)
    
    dct = {"west":-120,"south":30,"east":-70,"north":45}
    aoi2 = validate_aoi(dct)
    assert aoi2.bounds == (-120,30,-70,45)

def test_within_conus_true_false():
    aoi1 = box(-100, 30, -90, 40)
    aoi2 = box(-130, 20, -126, 23)
    assert within_conus(aoi1) is True
    assert within_conus(aoi2) is False
