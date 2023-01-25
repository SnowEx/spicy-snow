"""
Functions to search and download Sentinel-1 images for specific geometries and dates
"""

import asf_search
import pandas as pd
import shapely.geometry
import xarray

def s1_img_search(area: shapely.geometry.box, dates: (str, str)) -> pd.DataFrame:
    """
    find dates and url of Sentinel-1 overpasses

    Args:
    area: Bounding box of desired area to search within
    dates: Start and end date to search between

    Returns:
    granules: Dataframe of Sentinel-1 granule names to download.
    """
    pass

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