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

    # get results from asf_search in date range and geometry
    results = asf.geo_search(platform = [asf.PLATFORM.SENTINEL1], intersectsWith = area.wkt,\
        start = dates[0], end = dates[1], processingLevel = asf.PRODUCT_TYPE.GRD_HD)

    # check with 0 results.
    if len(results) == 0:
        raise ValueError("No search results found.")

    # create pandas dataframe from json result
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

    # if existing job name provided then don't submit and simply watch existing jobs.
    if existing_job_name:
        rtc_jobs = hyp3.find_jobs(name = existing_job_name)
        # if no running jobs then just return succeeded jobs
        if len(rtc_jobs.filter_jobs(succeeded = False, failed = False)) == 0:
            return rtc_jobs.filter_jobs(succeeded = True)
        # otherwise watch running jobs
        hyp3.watch(rtc_jobs.filter_jobs(succeeded = False, failed = False))
        # refresh with new successes and failures
        rtc_jobs = hyp3.refresh(rtc_jobs)
        # return successful jobs
        return rtc_jobs.filter_jobs(succeeded = True)

    # gather granules to submit to the hyp3 pipeline
    granules = search_results['properties.sceneName']

    # create a new hyp3 batch to hold submitted jobs
    rtc_jobs = sdk.Batch()
    for g in tqdm(granules, desc = 'Submitting s1 jobs'):
        # submit rtc jobs and ask for incidence angle map, in amplitude, @ 30 m resolution
        # https://hyp3-docs.asf.alaska.edu/using/sdk_api/#hyp3_sdk.hyp3.HyP3.submit_rtc_job
        rtc_jobs += hyp3.submit_rtc_job(g, name = job_name, include_inc_map = True,\
            scale = 'amplitude', dem_matching = False, resolution = 30)

    # warn user this may take a few hours for big jobs
    print(f'Watching {len(rtc_jobs)}. This may take a while...')

    # have hyp3 watch and update progress bar every 60 seconds
    hyp3.watch(rtc_jobs)

    # refresh jobs list with successes and failures
    rtc_jobs = hyp3.refresh(rtc_jobs)

    # filter out failed jobs
    failed_jobs = rtc_jobs.filter_jobs(succeeded=False, running=False, failed=True)
    if len(failed_jobs) > 0:
        print(f'{len(failed_jobs)} jobs failed.')
    
    # return only successful jobs
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
    # make data directory to store incoming tifs
    os.makedirs(outdir, exist_ok = True)
    # list to hold new DataArrays from downloaded tiffs
    das = []
    # list to check if a granule is repeated in the job list
    granules = []

    for job in tqdm(jobs, desc = 'Downloading S1 images'):
        # capture url from job description
        u = job.files[0]['url']
        # capture granule (for metadata scraping)
        granule = job.job_parameters['granules'][0]
        # skip this loop if granule is repeated in job list
        if granule in granules:
            continue
        # otherwise append to granules list
        granules.append(granule)
        # determine if scene is asc or des
        meta_results = asf.search(granule_list = [granule],
                                  flightDirection='Ascending')
        if bool(meta_results) == True:
            flight_dir = 'ascending'
        else:
            flight_dir = 'descending'
        # create dictionary to hold cloud url from .zip url
        # this lets us download only VV, VH, inc without getting other data from zip
        urls = {}
        urls[f'{granule}_VV'] = u.replace('.zip', '_VV.tif')
        urls[f'{granule}_VH'] = u.replace('.zip', '_VH.tif')
        urls[f'{granule}_inc'] = u.replace('.zip', '_inc_map.tif')
        # list to hold each band of image for concating to multi-band image
        imgs = []
        for name, url in urls.items():
            # download url to a tif file
            url_download(url, join(outdir, f'{name}.tif'), verbose = False)
            # open image in xarray
            img = rxa.open_rasterio(join(outdir, f'{name}.tif'))
            # reproject to WGS84
            img = img.rio.reproject('EPSG:4326')
            # clip to user specified area
            img = img.rio.clip([area], 'EPSG:4326')
            # add time as a indexable parameter
            img = img.assign_coords(time = pd.to_datetime(granule.split('_')[4]))
            # add flight direction as indexable parameter
            img = img.assign_coords(flight_dir = flight_dir)
            # add platform as indexable parameter
            platform = granule[0:3]
            img = img.assign_coords(platform = platform)
            # add relative orbit as indexable parameter
            # https://gis.stackexchange.com/questions/237116/sentinel-1-relative-orbit
            if platform == 'S1A':
                relative_orbit = ((int(granule[49:55])-73)%175)+1
            else:
                relative_orbit = ((int(granule[49:55])-27)%175)+1
            img = img.assign_coords(relative_orbit = relative_orbit)
            # create band name
            band_name = name.replace(f'{granule}_', '')
            # add band to image
            imgs.append(img.assign_coords(band = [band_name]))
        # concat VV, VH, and inc into one xarray DataArray
        da = xr.concat(imgs, dim = 'band')
        # we need to reproject each image to match the first image to make CRSs work
        if das:
            da = da.rio.reproject_match(das[0])
        # append multi-band image to das list to concat into time-series DataArray
        das.append(da)
    # take list of multi-band images with different time values and make time series
    full_da = xr.concat(das, dim = 'time')

    # remove temp directory of tiffs
    if clean:
        shutil.rmtree(outdir)
    # return the full DataArray of time series multi-band (vv, vh, inc) images clipped to region
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
    # submit asf_search results to the hyp3 pipeline and watch for jobs to run
    rtc_jobs = hyp3_pipeline(search_results = search_results, job_name = job_name, existing_job_name = existing_job_name)
    # download tiffs from successful hyp3 pipeline and convert to the xarray DataArray
    s1_dataArray = hyp3_jobs_to_dataArray(jobs = rtc_jobs, area = area, outdir = tmp_dir, clean = False)
    # promote to DataSet and set sentinel 1 image to 's1' data variable
    s1_dataset = s1_dataArray.to_dataset(name = 's1', promote_attrs = True)
    # save to netcdf for testing
    # s1_dataset.to_netcdf(out_fp)
    return s1_dataset

# End of file