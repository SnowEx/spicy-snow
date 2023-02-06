"""
Functions to search and download Sentinel-1 images for specific geometries and dates
"""

import sys
import os
from os.path import basename, exists, expanduser, join
import shutil
import asf_search as asf
import pandas as pd
import xarray as xr
import rioxarray as rxa
from rioxarray.merge import merge_arrays
import shapely.geometry
from datetime import date
from tqdm import tqdm
import hyp3_sdk as sdk
from hyp3_sdk.exceptions import AuthenticationError

sys.path.append(expanduser('~/Documents/spicy-snow'))
from spicy_snow.utils.download import url_download

def s1_img_search(area: shapely.geometry.box, dates: (str, str)) -> pd.DataFrame:
    """
    find dates and url of Sentinel-1 overpasses

    Args:
    area: Bounding box of desired area to search within
    dates: Start and end date to search between

    Returns:
    granules: Dataframe of Sentinel-1 granule names to download.
    """
    # Error Checking
    if len(dates) != 2:
        raise TypeError("Provide at start and end date in format (YYYY-MM-DD, YYYY_MM_DD)")
    if type(area) != shapely.geometry.polygon.Polygon:
        raise TypeError("Geometry must be a shapely.geometry.box type")
    if type(dates[0]) != str:
        raise TypeError("Provide at start and end date in format (YYYY-MM-DD, YYYY_MM_DD)")
    dates = [pd.to_datetime(d) for d in dates]
    if dates[1] < dates[0]:
        raise ValueError("End date is before start date")
    if dates[0].year < 2017 or dates[1].year < 2017:
        raise IndexError("Dates are prior to Sentinel-1 launch dates")
    if dates[0].date() > date.today() or dates[1].date() > date.today():
        raise IndexError("Dates are in the future.")
    if area.bounds[3] > 90 or area.bounds[1] < 0 or area.bounds[2] > 180\
        or area.bounds[0] < -180:
        raise IndexError("Coordinates must be between 0-90N and -180-180")

    # get results
    results = asf.geo_search(platform = [asf.PLATFORM.SENTINEL1], intersectsWith = area.wkt,\
        start = dates[0], end = dates[1], processingLevel = asf.PRODUCT_TYPE.GRD_HD)

    # TODO check with 0 results.
    if len(results) == 0:
        raise ValueError("No search results found.")

    results = pd.json_normalize(results.geojson(), record_path = ['features'])
    return results

def hyp3_pipeline(search_results: pd.DataFrame, job_name, existing_job_name = False) -> sdk.jobs.Batch:
    """
    Start and monitor Hyp3 pipeline for desired Sentinel-1 granules
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    search_results: Pandas Dataframe of asf_search search results.
    job_name: name to give hyp3 batch run
    existing_job_name: if you have an existing job that you want to find and reuse [default: False]

    Returns:
    rtc_jobs: Hyp3 batch object of completed jobs.
    """ 
    try:
        # .netrc
        hyp3 = sdk.HyP3()
    except AuthenticationError:
        # prompt for password
        hyp3 = sdk.HyP3(prompt = True)

    if existing_job_name:
        rtc_jobs = hyp3.find_jobs(name = existing_job_name)
        if len(rtc_jobs.filter_jobs(succeeded = False, failed = False)) > 0:
            hyp3.watch(rtc_jobs.filter_jobs(succeeded = False, failed = False))

        rtc_jobs = hyp3.refresh(rtc_jobs)
        return rtc_jobs.filter_jobs(succeeded = True)

    granules = search_results['properties.sceneName']

    rtc_jobs = sdk.Batch()
    for g in tqdm(granules, desc = 'Submitting Jobs'):
        # https://hyp3-docs.asf.alaska.edu/using/sdk_api/#hyp3_sdk.hyp3.HyP3.submit_rtc_job
        rtc_jobs += hyp3.submit_rtc_job(g, name = job_name, include_inc_map = True,\
            scale = 'amplitude', dem_matching = False, resolution = 30)

    print(f'Watching {len(rtc_jobs)}. This may take awhile...')
    hyp3.watch(rtc_jobs)

    rtc_jobs = hyp3.refresh(rtc_jobs)

    failed_jobs = rtc_jobs.filter_jobs(succeeded=False, running=False, failed=True)
    if len(failed_jobs) > 0:
        print(f'{len(failed_jobs)} jobs failed.')
    
    return rtc_jobs.filter_jobs(succeeded = True)

def hyp3_jobs_to_dataArray(jobs: sdk.jobs.Batch, area: shapely.geometry.box, outdir: str, clean = True) -> xr.DataArray:
    """
    Download rtc Sentinel-1 images from Hyp3 pipeline.
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    jobs: hyp3 Batch object of completed jobs
    outdir: directory to save tif files.
    clean: clean up tiffs after creating DataArray [default: True]

    Returns:
    da: DataArray of Sentinel VV+VH and incidence angle
    """
    os.makedirs(outdir, exist_ok = True)
    das = []
    granules = []
    for job in tqdm(jobs, desc = 'Downloading S1 to DataArray'):
        u = job.files[0]['url']
        granule = job.job_parameters['granules'][0]
        if granule in granules:
            continue
        granules.append(granule)

        urls = {}
        urls[f'{granule}_VV'] = u.replace('.zip', '_VV.tif')
        urls[f'{granule}_VH'] = u.replace('.zip', '_VH.tif')
        urls[f'{granule}_inc'] = u.replace('.zip', '_inc_map.tif')

        imgs = []
        for j, (name, url) in enumerate(urls.items()):
            url_download(url, join(outdir, f'{name}.tif'), verbose = False)
            img = rxa.open_rasterio(join(outdir, f'{name}.tif'))
            img = img.rio.reproject('EPSG:4326')
            img = img.rio.clip([area], 'EPSG:4326')
            band_name = name.replace(f'{granule}_', '')
            img = img.assign_coords(time = pd.to_datetime(granule.split('_')[4]))
            imgs.append(img.assign_coords(band = [band_name]))
        da = xr.concat(imgs, dim = 'band')
        if das:
            da = da.rio.reproject_match(das[0])
        das.append(da)

    full_da = xr.concat(das, dim = 'time')

    if clean:
        shutil.rmtree(outdir)

    return full_da

def download_s1_imgs(search_results: pd.DataFrame, area: shapely.geometry.box, job_name: str = 'sentinel-1-snow-depth', tmp_dir = './tmp', existing_job_name = False) -> xr.Dataset:
    """
    Download rtc Sentinel-1 images from Hyp3 pipeline.
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    search_results: Dataframe of asf_search Sentinel-1 granules to download
    job_name: job_name to use for hyp3 cloud processing. [default: 'sentinel-1-snow-depth]
    tmp_dir: temporary directory to save tifs to

    Returns:
    s1_dataset: Xarray dataset of Sentinel-1 backscatter and incidence angle
    """
    rtc_jobs = hyp3_pipeline(search_results = search_results, job_name = job_name, existing_job_name = existing_job_name)
    s1_dataArray = hyp3_jobs_to_dataArray(jobs = rtc_jobs, area = area, outdir = tmp_dir, clean = False)
    s1_dataset = s1_dataArray.to_dataset(name = 's1', promote_attrs = True)
    # s1_dataset.to_netcdf(out_fp)
    return s1_dataset

if __name__ == '__main__':
    area = shapely.geometry.box(-114.4, 43, -114.3, 43.1)
    dates = ('2019-12-28', '2020-02-02')
    search_results = s1_img_search(area, dates)
    # print(search_results)
    import pickle
    # with open('/Users/zachkeskinen/Documents/spicy-snow/tests/test_data/search_result.pkl', 'wb') as f:
    #     pickle.dump(search_results, f)
    da = download_s1_imgs(search_results, out_fp = '/Users/zachkeskinen/Documents/spicy-snow/contrib/keskinen/data/test.nc', job_name = 'test2-zk')
    print(da)
    with open('/Users/zachkeskinen/Documents/spicy-snow/tests/test_data/s1_da.pkl', 'wb') as f:
        pickle.dump(da, f)