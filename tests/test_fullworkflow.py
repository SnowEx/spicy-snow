import pytest

import os
from pathlib import Path
from spicy_snow.retrieval import retrieve_snow_depth
import xarray as xr

def run_full_workflow_opera():

    # change to your minimum longitude, min lat, max long, max lat
    test_aoi = [-114.5, 43, -114, 44]

    # can also be a point for point based retrievals
    # test_aoi = [-114.2, 43.75]

    # you will want to start before the snowfalls since this is a change detection method
    dates = ['2020-10-01', '2021-05-01']

    # this will be where your results are saved
    out_nc = Path('./local/test.nc').resolve()
    work_dir = Path('./local/').resolve()

    # resolution in meters to work at (100, 500, 1km all have been tested).
    # default is 100 meters
    spatial_resolution = 100

    # data source can be either "opera" RTC or "hyp3" cloud processed
    # opera is much faster and doesn't require hyp3 credits
    # hyp3 uses gamma processing and may be cleaner
    source = 'opera' # other option "hyp3"

    spicy_ds = retrieve_snow_depth(aoi = test_aoi, dates = dates, 
                                work_dir = work_dir, 
                                resolution = spatial_resolution,
                                debug=False,
                                source = source,
                                outfp=out_nc)
    
    ds = xr.open_dataset(out_nc)
    assert ds.time.size == 86
    assert ds.x.size == 423
    assert ds.y.size == 846
    assert 'vv' in ds.data_vars
    assert 'vh' in ds.data_vars
    assert 'snow_depth' in ds.data_vars
    assert 'fcf' in ds.data_vars
    assert 'snowcover' in ds.data_vars


def run_full_workflow_hyp3():
    """
    second run with hyp3
    use much small date range to avoid too long
    should be faster after first run through due to hyp3 caching
    """

    # change to your minimum longitude, min lat, max long, max lat
    test_aoi = [-114.5, 43, -114, 44]

    # can also be a point for point based retrievals
    # test_aoi = [-114.2, 43.75]

    # you will want to start before the snowfalls since this is a change detection method
    dates = ['2021-12-01', '2021-12-10']

    # this will be where your results are saved
    out_nc = Path('./local/test_hyp3.nc').resolve()
    work_dir = Path('./local/').resolve()

    # resolution in meters to work at (100, 500, 1km all have been tested).
    # default is 100 meters
    spatial_resolution = 100

    source = 'hyp3'
    spicy_ds = retrieve_snow_depth(aoi = test_aoi, dates = dates, 
                                work_dir = work_dir, 
                                resolution = spatial_resolution,
                                debug=False,
                                source = source,
                                job_name='testing_hyp3_spicy',
                                outfp=out_nc)
    
    ds = xr.open_dataset(out_nc)
    assert 'vv' in ds.data_vars
    assert 'vh' in ds.data_vars

@pytest.mark.integration
def test_readme_workflow_opera():
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run integration tests")

    run_full_workflow_opera()

@pytest.mark.integration
def test_readme_workflow_hyp3():
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run integration tests")

    run_full_workflow_hyp3()
