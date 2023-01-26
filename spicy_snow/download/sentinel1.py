"""
Functions to search and download Sentinel-1 images for specific geometries and dates
"""

import sys
import os
from os.path import basename, exists, expanduser, join
import asf_search as asf
import pandas as pd
import xarray as xr
import rioxarray as rxa
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
    # TODO Error Checking
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
    results = pd.json_normalize(results.geojson(), record_path = ['features'])
    return results

def download_s1_imgs(search_results: pd.DataFrame, out_dir: str, job_name: str = 'zk-s1-snow') -> xr.DataArray:
    """
    Download rtc Sentinel-1 images from Hyp3 pipeline.
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    search_results: Dataframe of asf_search Sentinel-1 granules to download

    Returns:
    s1_dataset: Xarray dataset of Sentinel-1 backscatter and incidence angle
    """
    try:
        # .netrc
        hyp3 = sdk.HyP3()
    except AuthenticationError:
        # prompt for password
        hyp3 = sdk.HyP3(prompt = True)

    granules = search_results['properties.sceneName']

    rtc_jobs = sdk.Batch()
    for g in granules:
        # https://hyp3-docs.asf.alaska.edu/using/sdk_api/#hyp3_sdk.hyp3.HyP3.submit_rtc_job
        rtc_jobs += hyp3.submit_rtc_job(g, name = job_name, include_inc_map = True,\
            scale = 'amplitude', dem_matching = False, resolution = 30)
    hyp3.watch(rtc_jobs)

    failed_jobs = rtc_jobs.filter_jobs(succeeded=False, running=False, failed=True)
    if len(failed_jobs) > 0:
        print(f'Some jobs failed. Number of failed jobs: {len(failed_jobs)}')
    
    das = []
    for i, job in enumerate(rtc_jobs[:2]):
        u = job.files[0]['url']
        granule = job.job_parameters['granules'][0]
        urls = {}
        urls[f'{granule}_VV'] = u.replace('.zip', '_VV.tif')
        urls[f'{granule}_VH'] = u.replace('.zip', '_VH.tif')
        urls[f'{granule}_inc'] = u.replace('.zip', '_inc_map.tif')
        for j, (name, url) in enumerate(urls.items()):
            url_download(url, join(out_dir,f'{name}.tif'))
            img = rxa.open_rasterio(join(out_dir,f'{name}.tif').rio.clip([area], 'EPSG:4326'))
            img = img.rio.reproject('EPSG:4326')
            band_name = name.replace(f'{granule}_', '')
            img = img.assign_coords(time = pd.to_datetime(granule.split('_')[4]))
            if j == 0:
                da = img.assign_coords(band = [band_name])
                da.attrs["long_name"] = granule
                da.attrs["mission"] = granule.split('_')[0]
                da.attrs["mode"] = granule.split('_')[1]
                da.attrs["product-type"] = granule.split('_')[2][:3]
                da.attrs["resolution"] = granule.split('_')[2][3]
                da.attrs["processing-level"] = granule.split('_')[3][0]
                da.attrs["product-class"] = granule.split('_')[3][1]
                da.attrs["polarization-type"] = granule.split('_')[3][2:]
                da.attrs["start-time"] = pd.to_datetime(granule.split('_')[4])
                da.attrs["end-time"] = pd.to_datetime(granule.split('_')[5])
                da.attrs["absolute-orbit"] = granule.split('_')[6]
                da.attrs["take-id"] = granule.split('_')[7]
                da.attrs["unique-id"] = granule.split('_')[8]
            else:
                da = xr.concat([da, img.assign_coords(band = [band_name])], dim = 'band')
        das.append(da)
    da = xr.concat(das, dim = 'time')
    return da


if __name__ == '__main__':
    area = shapely.geometry.box(-114.4, 43, -114.3, 43.1)
    dates = ('2019-12-28', '2020-02-02')
    search_results = s1_img_search(area, dates)
    # print(search_results)
    # import pickle
    # with open('/Users/zachkeskinen/Documents/spicy-snow/tests/test_data/search_result.pkl', 'wb') as f:
    #     pickle.dump(search_results, f)
    da = download_s1_imgs(search_results, out_dir = '/Users/zachkeskinen/Documents/spicy-snow/contrib/keskinen/data')
    print(da)

