"""
Main user function to retrieve snow depth with snow depth and wet snow flag
"""
import os
from os.path import join
import xarray as xr
import shapely.geometry
from typing import Tuple, Union
import logging

# Add main repo to path
import sys
from os.path import expanduser
sys.path.append(expanduser('../'))

# import functions for downloading
from spicy_snow.download.sentinel1 import s1_img_search, hyp3_pipeline, download_hyp3, combine_s1_images
from spicy_snow.download.forest_cover import download_fcf
from spicy_snow.download.snow_cover import download_snow_cover

# import functions for pre-processing
from spicy_snow.processing.s1_preprocessing import merge_partial_s1_images, s1_orbit_averaging,\
s1_clip_outliers, subset_s1_images, ims_water_mask, s1_incidence_angle_masking, merge_s1_subsets

# import the functions for snow_index calculation
from spicy_snow.processing.snow_index import calc_delta_VV, calc_delta_cross_ratio, \
    calc_delta_gamma, clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth

# import the functions for wet snow flag
from spicy_snow.processing.wet_snow import id_newly_frozen_snow, id_newly_wet_snow, \
    id_wet_negative_si, flag_wet_snow

# setup root logger
from spicy_snow.utils.spicy_logging import setup_logging

def retrieve_snow_depth(area: shapely.geometry.Polygon, 
                        dates: Tuple[str, str], 
                        work_dir: str = './',
                        job_name: str = 'spicy-snow-run',
                        existing_job_name: Union[bool, str] = False,
                        debug: bool = False,
                        outfp: Union[str, bool] = False) -> xr.Dataset:
    """
    Finds, downloads Sentinel-1, forest cover, water mask, snow coverage. Then retrieves snow depth
    using Lievens et al. 2021 method.

    Args:
    area: Bounding box of desired area to search within
    dates: Start and end date to search between
    work_dir: filepath to directory to work in. Will be created if not existing
    job_name: name for hyp3 job
    existing_job_name: name for preexisiting hyp3 job to download and avoid resubmitting
    debug: do you want to get verbose logging?
    outfp: do you want to save netcdf? default is False and will just return dataset

    Returns:
    datset: Xarray dataset with 'snow_depth' and 'wet_snow' variables for all Sentinel-1
    image acquistions in area and dates
    """
    os.makedirs(work_dir , exist_ok = True)

    setup_logging(log_dir = join(work_dir, 'logs'), debug = debug)
    log = logging.getLogger(__name__)

    ## Downloading Steps

    # get asf_search search results
    search_results = s1_img_search(area, dates)
    log.info(f'Found {len(search_results)} results')

    # download s1 images into dataset ['s1'] keyword
    jobs = hyp3_pipeline(search_results, job_name = job_name, existing_job_name = existing_job_name)
    imgs = download_hyp3(jobs, area, outdir = join(work_dir, 'tmp'), clean = False)
    ds = combine_s1_images(imgs)

    # merge partial images together
    ds = merge_partial_s1_images(ds)

    # download IMS snow cover and add to dataset ['ims'] keyword
    ds = download_snow_cover(ds, tmp_dir = join(work_dir, 'tmp'), clean = False)

    # download fcf and add to dataset ['fcf'] keyword
    ds = download_fcf(ds, join(work_dir, 'tmp', 'fcf.tif'))

    ## Preprocessing Steps
    log.info("Preprocessing Sentinel-1 images")
    # subset dataset by flight_dir and platform
    dict_ds = subset_s1_images(ds)

    for subset_name, subset_ds in dict_ds.items():
        # average each orbit to overall mean
        dict_ds[subset_name] = s1_orbit_averaging(subset_ds)
        # clip outlier values of backscatter to overall mean
        dict_ds[subset_name] = s1_clip_outliers(subset_ds)
    
    # recombine subsets
    ds = merge_s1_subsets(dict_ds)

    # add water mask
    # ds = ims_water_mask(ds)

    # mask out outliers in incidence angle
    ds = s1_incidence_angle_masking(ds)

    ## Snow Index Steps
    log.info("Calculating snow index")
    # calculate delta CR and delta VV
    ds = calc_delta_cross_ratio(ds)
    ds = calc_delta_VV(ds)

    # calculate delta gamma with delta CR and delta VV with FCF
    ds = calc_delta_gamma(ds)

    # clip outliers of delta gamma
    ds = clip_delta_gamma_outlier(ds)

    # calculate snow_index from delta_gamma
    ds = calc_snow_index(ds)

    # convert snow index to snow depth
    ds = calc_snow_index_to_snow_depth(ds)

    ## Wet Snow Flags
    log.info("Flag wet snow")
    # find newly wet snow
    ds = id_newly_wet_snow(ds)
    ds = id_wet_negative_si(ds)

    # find newly frozen snow
    ds = id_newly_frozen_snow(ds)

    # make wet_snow flag
    ds = flag_wet_snow(ds)

    if outfp:
        outfp = str(outfp)
        
        ds.to_netcdf(outfp)

    return ds

