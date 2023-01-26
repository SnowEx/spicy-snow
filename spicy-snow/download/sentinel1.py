"""
Functions to search and download Sentinel-1 images for specific geometries and dates
"""

import asf_search as asf
import pandas as pd
import shapely.geometry
import xarray
from datetime import date

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

def download_s1_imgs(granules: pd.DataFrame) -> xarray:
    """
    Download rtc Sentinel-1 images from Hyp3 pipeline.
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    granules: Dataframe of Sentinel-1 granule names to download

    Returns:
    s1_dataset: Xarray dataset of Sentinel-1 backscatter and incidence angle
    """
    pass

if __name__ == '__main__':
    area = shapely.geometry.box(-114.4, 43, -114.3, 43.1)
    dates = ('2019-12-28', '2020-02-02')
    search_results = s1_img_search(area, dates)
    print(search_results)
    import pickle
    with open('/Users/zachkeskinen/Documents/spicy-snow/tests/test_data/search_result.pkl', 'wb') as f:
        pickle.dump(search_results, f)
