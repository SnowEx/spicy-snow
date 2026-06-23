"""
Integration test for the full retrieve_snow_depth pipeline.

Based on scripts/readme_run.py — uses the Sun Valley / Sawtooth area
with a short date range to keep download and processing times manageable.

Requires network access and Earthdata / ASF credentials.

Run explicitly:
    pytest tests/integration/ -v -m integration
Skip in normal test runs:
    pytest -m "not integration"
"""
import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from spicy_snow.retrieval import retrieve_snow_depth


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AOI = [-114.5, 43.5, -114.25, 43.75]
DATES = ("2020-10-01", "2021-05-01")
SPATIAL_RESOLUTION = 500

LOCAL_CACHE = Path(__file__).resolve().parents[2] / "local"

EXPECTED_DATA_VARS = [
    "vv",
    "vh",
    "snow_depth",
    "snow_index",
    "wet_snow",
    "snowcover",
    "fcf",
]

EXPECTED_INTERMEDIATE_VARS = [
    "deltavv",
    "deltaCR",
    "deltaGamma",
]


# ---------------------------------------------------------------------------
# Fixture — run the pipeline once for all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snow_depth_result(tmp_path_factory):
    """Run the full snow depth retrieval pipeline once per module."""
    work_dir = tmp_path_factory.mktemp("spicy_integration")
    out_nc = work_dir / "test_integration.nc"

    ds = retrieve_snow_depth(
        aoi=AOI,
        dates=DATES,
        work_dir=str(work_dir),
        resolution=SPATIAL_RESOLUTION,
        debug=False,
        outfp=str(out_nc),
    )
    return ds, out_nc


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_returns_xarray_dataset(snow_depth_result):
    ds, _ = snow_depth_result
    assert isinstance(ds, xr.Dataset)


@pytest.mark.integration
def test_expected_data_vars_present(snow_depth_result):
    ds, _ = snow_depth_result
    missing = [v for v in EXPECTED_DATA_VARS if v not in ds.data_vars]
    assert missing == [], f"Missing data variables: {missing}"


@pytest.mark.integration
def test_expected_intermediate_vars_present(snow_depth_result):
    ds, _ = snow_depth_result
    missing = [v for v in EXPECTED_INTERMEDIATE_VARS if v not in ds.data_vars]
    assert missing == [], f"Missing intermediate variables: {missing}"


@pytest.mark.integration
def test_dataset_has_time_dim(snow_depth_result):
    ds, _ = snow_depth_result
    assert "time" in ds.dims
    assert ds.sizes["time"] > 0


@pytest.mark.integration
def test_dataset_has_spatial_dims(snow_depth_result):
    ds, _ = snow_depth_result
    assert "y" in ds.dims
    assert "x" in ds.dims
    assert ds.sizes["y"] > 0
    assert ds.sizes["x"] > 0


@pytest.mark.integration
def test_dataset_dims_order(snow_depth_result):
    """Dataset should be transposed to (time, y, x) order."""
    ds, _ = snow_depth_result
    assert list(ds["snow_depth"].dims) == ["time", "y", "x"]


# ---------------------------------------------------------------------------
# Snow depth: range and finiteness
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_snow_depth_has_finite_values(snow_depth_result):
    ds, _ = snow_depth_result
    assert np.isfinite(ds["snow_depth"].values).any(), "snow_depth is entirely NaN/Inf"


@pytest.mark.integration
def test_snow_depth_nonnegative(snow_depth_result):
    """Snow depth should be >= 0 everywhere it is defined."""
    ds, _ = snow_depth_result
    finite = ds["snow_depth"].values[np.isfinite(ds["snow_depth"].values)]
    assert finite.min() >= 0.0, f"snow_depth has negative values: min={finite.min()}"


@pytest.mark.integration
def test_snow_depth_reasonable_max(snow_depth_result):
    """Snow depth should not exceed 20 meters anywhere."""
    ds, _ = snow_depth_result
    finite = ds["snow_depth"].values[np.isfinite(ds["snow_depth"].values)]
    assert finite.max() <= 20.0, f"snow_depth unreasonably large: max={finite.max()}"


# ---------------------------------------------------------------------------
# Snow index
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_snow_index_nonnegative(snow_depth_result):
    ds, _ = snow_depth_result
    finite = ds["snow_index"].values[np.isfinite(ds["snow_index"].values)]
    assert finite.min() >= 0.0, f"snow_index has negative values: min={finite.min()}"


@pytest.mark.integration
def test_snow_index_has_variation(snow_depth_result):
    """Snow index should not be constant across all pixels and times."""
    ds, _ = snow_depth_result
    finite = ds["snow_index"].values[np.isfinite(ds["snow_index"].values)]
    assert finite.std() > 0, "snow_index has zero variance"


# ---------------------------------------------------------------------------
# Wet snow flags
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_wet_snow_is_binary(snow_depth_result):
    ds, _ = snow_depth_result
    values = ds["wet_snow"].values
    finite = values[np.isfinite(values)]
    unique = np.unique(finite)
    assert set(unique).issubset({0.0, 1.0}), f"wet_snow has non-binary values: {unique}"


# ---------------------------------------------------------------------------
# Forest cover fraction
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fcf_bounded_zero_one(snow_depth_result):
    ds, _ = snow_depth_result
    finite = ds["fcf"].values[np.isfinite(ds["fcf"].values)]
    assert finite.min() >= 0.0, f"fcf below 0: {finite.min()}"
    assert finite.max() <= 1.0, f"fcf above 1: {finite.max()}"


# ---------------------------------------------------------------------------
# Backscatter
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_vv_has_finite_values(snow_depth_result):
    ds, _ = snow_depth_result
    assert np.isfinite(ds["vv"].values).any(), "vv is entirely NaN/Inf"


@pytest.mark.integration
def test_vh_has_finite_values(snow_depth_result):
    ds, _ = snow_depth_result
    assert np.isfinite(ds["vh"].values).any(), "vh is entirely NaN/Inf"


@pytest.mark.integration
def test_vv_in_db_range(snow_depth_result):
    """VV backscatter in dB should be in a reasonable range."""
    ds, _ = snow_depth_result
    finite = ds["vv"].values[np.isfinite(ds["vv"].values)]
    assert finite.min() > -50, f"vv suspiciously low: {finite.min()}"
    assert finite.max() < 20, f"vv suspiciously high: {finite.max()}"


# ---------------------------------------------------------------------------
# Dataset attributes
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_param_attributes_set(snow_depth_result):
    ds, _ = snow_depth_result
    assert "param_A" in ds.attrs
    assert "param_B" in ds.attrs
    assert "param_C" in ds.attrs


@pytest.mark.integration
def test_bounds_attribute_set(snow_depth_result):
    ds, _ = snow_depth_result
    assert "bounds" in ds.attrs


# ---------------------------------------------------------------------------
# NetCDF output
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_netcdf_output_exists(snow_depth_result):
    _, out_nc = snow_depth_result
    assert out_nc.exists(), f"NetCDF not found at {out_nc}"
    assert out_nc.stat().st_size > 0, "NetCDF is empty"


@pytest.mark.integration
def test_netcdf_roundtrip(snow_depth_result):
    """Loading the saved NetCDF should produce the same variables."""
    ds_original, out_nc = snow_depth_result
    ds_loaded = xr.open_dataset(out_nc)
    for var in EXPECTED_DATA_VARS:
        assert var in ds_loaded.data_vars, f"{var} missing from loaded NetCDF"
    ds_loaded.close()
