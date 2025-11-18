import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from unittest.mock import patch, MagicMock
from pathlib import Path

import spicy_snow.find_data as fd

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def example_aoi():
    # AOI as [xmin, ymin, xmax, ymax]
    return [-110, 40, -105, 45]

@pytest.fixture
def example_results_df():
    data = {
        'properties.url': ['https://example.com/file1.tif', 'https://example.com/file2.tif'],
        'properties.additionalUrls': [['https://example.com/file1_aux.tif'], []],
        'properties.pathNumber': [1, 2],
        'properties.flightDirection': ['ASCENDING', 'DESCENDING'],
        'properties.polarization': ['VV', 'VH'],
        'properties.startTime': ['2025-01-01T00:00:00', '2025-01-02T00:00:00'],
        'properties.stopTime': ['2025-01-01T01:00:00', '2025-01-02T01:00:00'],
        'properties.sceneName': ['scene1', 'scene2'],
        'geometry.coordinates': [[[[0,0],[0,1],[1,1],[1,0],[0,0]]], [[[0,0],[0,1],[1,1],[1,0],[0,0]]]]
    }
    df = pd.DataFrame(data)
    return df

# -----------------------------
# Tests for find_snowcover_urls
# -----------------------------
@patch("spicy_snow.find_data.earthaccess.search_data")
def test_find_snowcover_urls(mock_search, example_aoi):
    # Setup mock
    mock_result = MagicMock()
    mock_result.data_links.return_value = ['https://example.com/test.A2025001.v1.tif', 'https://example.com/test.A2025002.v1.tif']
    mock_search.return_value = [mock_result]

    urls = fd.find_snowcover_urls(example_aoi, "2025-01-01", "2025-01-02")
    assert urls == ['https://example.com/test.A2025001.v1.tif', 'https://example.com/test.A2025002.v1.tif']

    # Test filtering by date_list
    urls_filtered = fd.find_snowcover_urls(example_aoi, "2025-01-01", "2025-01-02", date_list=["2025-01-01"])
    assert urls_filtered == ['https://example.com/test.A2025001.v1.tif']

# -----------------------------
# Tests for get_urls_from_asf_search
# -----------------------------
def test_get_urls_from_asf_search(example_results_df):
    urls = fd.get_urls_from_asf_search(example_results_df)
    expected_urls = [
        'https://example.com/file1.tif',
        'https://example.com/file1_aux.tif',
        'https://example.com/file2.tif'
    ]
    assert urls == expected_urls

# -----------------------------
# Tests for subset_asf_search_results
# -----------------------------
def test_subset_asf_search_results_aoi(example_results_df):
    # Use AOI intersecting everything (simplified)
    result = fd.subset_asf_search_results(example_results_df, aoi=box(-1, -1, 2, 2))
    assert len(result) == 2

def test_subset_asf_search_results_filters(example_results_df):
    # Path number filter
    result = fd.subset_asf_search_results(example_results_df, path_numbers=[1])
    assert all(result['properties.pathNumber'] == 1)

    # Direction filter
    result = fd.subset_asf_search_results(example_results_df, direction='DESCENDING')
    assert all(result['properties.flightDirection'] == 'DESCENDING')

    # Polarization filter
    result = fd.subset_asf_search_results(example_results_df, polarization='VH')
    assert all(result['properties.polarization'] == 'VH')

    # Start/stop time filter
    result = fd.subset_asf_search_results(example_results_df, start_time="2025-01-02")
    assert all(pd.to_datetime(result['properties.startTime']) >= pd.to_datetime("2025-01-02"))

    result = fd.subset_asf_search_results(example_results_df, stop_time="2025-01-01")
    assert all(pd.to_datetime(result['properties.stopTime']) <= pd.to_datetime("2025-01-01"))

    # Scene name filter
    result = fd.subset_asf_search_results(example_results_df, scene_name="scene1")
    assert all(result['properties.sceneName'] == "scene1")

# -----------------------------
# Tests for download_proba_v
# -----------------------------
@patch("spicy_snow.find_data.download_url")
def test_download_proba_v(mock_download):
    mock_download.return_value = "/tmp/forest_cover.tif"
    out_fp = fd.download_proba_v("/tmp/forest_cover.tif")
    assert out_fp == "/tmp/forest_cover.tif"
    mock_download.assert_called_once()
