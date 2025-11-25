import pytest
import warnings
from datetime import datetime, timedelta, date
from shapely.geometry import box, Polygon

from spicy_snow.utils.checks import (
    validate_urls,
    validate_dates,
    validate_aoi,
    within_conus
)

# ----------------------------
# validate_urls tests
# ----------------------------

def test_validate_urls_flatten_and_strip():
    urls = [" http://example.com ", ["https://test.com", None], "ftp://ftp.test.org"]
    result = validate_urls(urls)
    assert "http://example.com" in result
    assert "https://test.com" in result
    assert "ftp://ftp.test.org" in result
    assert None not in result

def test_validate_urls_invalid_url_warns():
    urls = ["not_a_url"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_urls(urls)
        assert any("does not look valid" in str(wi.message) for wi in w)
        assert result == ["not_a_url"]

def test_validate_urls_no_valid_raises():
    with pytest.raises(ValueError):
        validate_urls([None, "", "  "])

def test_validate_urls_non_string_warns():
    urls = ["http://valid.com", 123, {"url": "test"}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_urls(urls)
        assert any("Skipping non-string URL" in str(wi.message) for wi in w)
        assert "http://valid.com" in result

# ----------------------------
# validate_dates tests
# ----------------------------

def test_validate_dates_normal():
    start = "2015-01-01"
    end = "2015-12-31"
    s, e = validate_dates(start, end)
    assert s.year == 2015
    assert e.year == 2015

def test_validate_dates_future_error():
    future = date.today() + timedelta(days=1)
    with pytest.raises(ValueError):
        validate_dates("2015-01-01", future)

def test_validate_dates_start_after_end_error():
    with pytest.raises(ValueError):
        validate_dates("2020-01-01", "2019-12-31")

def test_validate_dates_s1b_warning():
    # End date intersects Sentinel-1B outage
    start = "2022-01-01"
    end = "2022-06-01"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_dates(start, end)
        assert any("Sentinel-1B outage" in str(wi.message) for wi in w)

def test_validate_dates_missing_start_or_end():
    with pytest.raises(ValueError):
        validate_dates(None, "2020-01-01")
    with pytest.raises(ValueError):
        validate_dates("2020-01-01", None)

# ----------------------------
# validate_aoi tests
# ----------------------------

def test_validate_aoi_list_and_box():
    coords = [-100, 30, -90, 40]
    geom = validate_aoi(coords)
    assert isinstance(geom, Polygon)
    assert geom.bounds == tuple(coords)

def test_validate_aoi_dict_keys():
    aoi_dict = {"xmin": -100, "ymin": 30, "xmax": -90, "ymax": 40}
    geom = validate_aoi(aoi_dict)
    assert geom.bounds == (-100, 30, -90, 40)

    aoi_dict2 = {"west": -120, "south": 20, "east": -110, "north": 25}
    geom = validate_aoi(aoi_dict2)
    assert geom.bounds == (-120, 20, -110, 25)

def test_validate_aoi_auto_fix_reverse():
    aoi_dict = {"xmin": -90, "ymin": 40, "xmax": -100, "ymax": 30}
    geom = validate_aoi(aoi_dict)
    assert geom.bounds == (-100, 30, -90, 40)

def test_validate_aoi_polygon_passthrough():
    poly = box(-120, 20, -110, 30)
    result = validate_aoi(poly)
    assert result.equals(poly)

def test_validate_aoi_invalid_input():
    with pytest.raises(ValueError):
        validate_aoi("invalid")

    with pytest.raises(ValueError):
        validate_aoi({"a": 1, "b": 2})

# ----------------------------
# within_conus tests
# ----------------------------

def test_within_conus_true():
    aoi = [-100, 30, -90, 40]
    assert within_conus(aoi) is True

def test_within_conus_false_outside():
    aoi = [-130, 10, -126, 15]  # far outside CONUS
    assert within_conus(aoi) is False

def test_within_conus_partial_overlap():
    aoi = [-130, 30, -100, 40]  # partially overlaps CONUS
    assert within_conus(aoi) is True
