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

from typing import Dict

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
    while existing_job_name:
        rtc_jobs = hyp3.find_jobs(name = existing_job_name)
        rtc_jobs = rtc_jobs.filter_jobs(succeeded = True, failed = False, \
            running = True, include_expired = False)

        # if no jobs found go to original search with name.
        if len(rtc_jobs) == 0:
            break

        # if no running jobs then just return succeeded jobs
        if len(rtc_jobs.filter_jobs(succeeded = False, failed = False)) == 0:
            return rtc_jobs.filter_jobs(succeeded = True)

        # otherwise watch running jobs
        hyp3.watch(rtc_jobs)

        # refresh with new successes and failures
        rtc_jobs = hyp3.refresh(rtc_jobs)
        
        # return successful jobs
        return rtc_jobs.filter_jobs(succeeded = True)
    
    # check if you have passed quota
    quota = hyp3.check_quota()
    if not quota or len(search_results) > hyp3.check_quota():
        print(f'More search results ({len(searh_results)}) than quota ({quota}).')
        resp = None
        while resp not in ['Y', 'N']:
            resp = input('Continue anyways?')[:1].upper()
            if resp not in ['Y', 'N']:
                print('Enter Y or N.')
        
        if resp == 'N':
            sys.exit("Not enough jobs left in ASF Hyp3 quota.")


    # gather granules to submit to the hyp3 pipeline
    granules = search_results['properties.sceneName']

    # create a new hyp3 batch to hold submitted jobs
    rtc_jobs = sdk.Batch()
    for g in tqdm(granules, desc = 'Submitting s1 jobs'):
        # submit rtc jobs and ask for incidence angle map, in dBs, @ 30 m resolution
        # https://hyp3-docs.asf.alaska.edu/using/sdk_api/#hyp3_sdk.hyp3.HyP3.submit_rtc_job
        rtc_jobs += hyp3.submit_rtc_job(g, name = job_name, include_inc_map = True,\
            scale = 'decibel', dem_matching = False, resolution = 30)

    # warn user this may take a few hours for big jobs
    print(f'Watching {len(rtc_jobs)} jobs. This may take a while...')

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

def download_hyp3(jobs: sdk.jobs.Batch, area: shapely.geometry.box, outdir: str, clean = True) -> Dict[str, xr.DataArray]:
    """
    Download rtc Sentinel-1 images from Hyp3 pipeline.
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    jobs: hyp3 Batch object of completed jobs
    outdir: directory to save tif files.
    clean: clean up tiffs after creating DataArray [default: True]

    Returns:
    images: dictionary of granule names and DataArrays
    """
    # make data directory to store incoming tifs
    os.makedirs(outdir, exist_ok = True)

    # results dictionary to send to next step
    dataArrays = {}

    # grab first granule name for reprojecting matching
    first_granule = jobs[0].job_parameters['granules'][0]

    # loop through jobs
    for job in tqdm(jobs, desc = 'Downloading S1 images'):
        # capture url from job description
        u = job.files[0]['url']

        # capture granule (for metadata scraping)
        granule = job.job_parameters['granules'][0]

        # skip this loop if granule is repeated in job list
        if granule in dataArrays.keys():
            continue

        # create dictionary to hold cloud url from .zip url
        # this lets us download only VV, VH, inc without getting other data from zip
        urls = {}
        urls[f'{granule}_VV'] = u.replace('.zip', '_VV.tif')
        urls[f'{granule}_VH'] = u.replace('.zip', '_VH.tif')
        urls[f'{granule}_inc'] = u.replace('.zip', '_inc_map.tif')
        imgs = []
        for name, url in urls.items():
            # download url to a tif file
            url_download(url, join(outdir, f'{name}.tif'), verbose = False)

            # open image in xarray
            img = rxa.open_rasterio(join(outdir, f'{name}.tif'), masked = True)

            # reproject to WGS84
            img = img.rio.reproject('EPSG:4326')

            # clip to user specified area
            img = img.rio.clip([area], 'EPSG:4326')

            # create band name
            band_name = name.replace(f'{granule}_', '')

            # add band to image
            img = img.assign_coords(band = [band_name])

            # add named band image to 3 image stack
            imgs.append(img)

        # concat VV, VH, and inc into one xarray DataArray
        da = xr.concat(imgs, dim = 'band')

        # we need to reproject each image to match the first image to make CRSs work
        if dataArrays:
            da = da.rio.reproject_match(dataArrays[first_granule])

        # add img to downloaded dataArrays list with granule as key
        dataArrays[granule] = da
    
    # remove temp directory of tiffs
    if clean:
        shutil.rmtree(outdir)

    return dataArrays

def combine_s1_images(dataArrays: Dict[str, xr.DataArray]) -> xr.Dataset:
    """
    Combine list of 3-banded Sentinel 1 data Arrays into a single xarray
    Dataset with associated metadata bands and attributes.

    Args:
    dataArrays: dictionary of granule name and 3 band Sentinel-1 data Arrays

    Returns:
    dataset: xr dataset with time dimension added, metadata attributes, and 
    metadata coordinates (flight direction, orbit #, platform)
    """
    das = []

    for granule, da in tqdm(dataArrays.items(), desc = 'Combining Sentinel-1 dataArrays'):
        # get granule metadata
        granule_metadata = asf.product_search(f'{granule}-GRD_HD')[0]

        # set flight direction
        flight_dir = granule_metadata.properties['flightDirection'].lower()

        # set relative orbit 
        relative_orbit = granule_metadata.properties['pathNumber']

        # expand time dimension of DataArray from zero dimension (scalar) to 1d
        da = da.expand_dims(dim = {'time': 1})

        # add time as a indexable parameter
        da = da.assign_coords(time = [pd.to_datetime(granule.split('_')[4])])

        # add flight direction as indexable parameter
        da = da.assign_coords(flight_dir = ('time', [flight_dir]))

        # add platform as indexable parameter
        platform = granule[0:3]
        da = da.assign_coords(platform = ('time', [platform]))

        # add relative orbit as indexable parameter
        da = da.assign_coords(relative_orbit = ('time', [relative_orbit]))
        
        # append multi-band image to das list to concat into time-series DataArray
        das.append(da)

    # take list of multi-band images with different time values and make time series
    s1_dataArray = xr.concat(das, dim = 'time')

    # make sentinel 1 dataset 
    s1_dataset = s1_dataArray.to_dataset(name = 's1', promote_attrs = True)

    # s1_units tag
    s1_dataset.attrs['s1_units'] = 'dB'

    # resolution:
    s1_dataset.attrs['resolution'] = '30'

    return s1_dataset

# End of file