"""
Main user function to retrieve snow depth with snow depth and wet snow flag
"""
import os
from os.path import join
from pathlib import Path
import pandas as pd
import xarray as xr
import shapely.geometry
from typing import Tuple, Union, List
import logging

# Add main repo to path
import sys
from os.path import expanduser
sys.path.append(expanduser('../'))

# import functions for downloading
from spicy_snow.find_data import get_sentinel1_urls, find_snowcover_urls
from spicy_snow.processing.generate_dataarrays import generate_sentinel1_dataarray,\
    generate_snowcover_dataarray, generate_forest_fraction_dataarray, convert_snowcover_dates_to_s1_overpasses

# import functions for pre-processing
from spicy_snow.processing.s1_preprocessing import s1_orbit_averaging, s1_clip_outliers, ims_water_mask, amplitude_to_dB

# import the functions for snow_index calculation
from spicy_snow.processing.snow_index import calc_delta_vv, calc_delta_cross_ratio, \
    calc_delta_gamma, clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth

# import the functions for wet snow flag
from spicy_snow.processing.wet_snow import id_newly_frozen_snow, id_newly_wet_snow, \
    id_wet_negative_si, flag_wet_snow

# setup root logger
from spicy_snow.utils.spicy_logging import setup_logging
from spicy_snow.utils.checks import validate_aoi, validate_dates
from spicy_snow.utils.download import download_urls, download_urls_parallel

def retrieve_snow_depth(aoi: shapely.geometry.Polygon, 
                        dates: Tuple[str, str], 
                        work_dir: str = './',
                        work_stem = None,
                        resolution = 100,
                        debug: bool = False,
                        ims_masking: bool = True,
                        wet_snow_thresh: float = -2,
                        freezing_snow_thresh: float = 1,
                        wet_SI_thresh: float = 0,
                        outfp: Union[str, Path, None] = None,
                        params: List[float] = [2.5, 0.2, 0.55]) -> xr.Dataset:
    """
    Finds, downloads Sentinel-1, forest cover, water mask (not implemented), and 
    snow coverage. Then retrieves snow depth using Lievens et al. 2021 method.

    Args:
    aoi: Shapely bounding box or [xmin, ymin, xmax, ymax] iterable of desired area to search within
    dates: Start and end date to search between
    work_dir: filepath to directory to work in. Will be created if not existing
    job_name: [Optional] Name for project file stems otherwise generated from dates
    debug: do you want to get verbose logging?
    ims_masking: do you want to mask pixels by IMS snow free imagery?
    wet_snow_thresh: what threshold in dB change to use for melting and re-freezing snow? Default: -2
    freezing_snow_thresh: what threshold in dB change to use for re-freezing snow id. Default: +2
    wet_SI_thresh: what threshold to use for negative snow index? Default: 0
    outfp: do you want to save netcdf? default is False and will just return dataset
    params: the A, B, C parameters to use in the model. Current defaults are optimized to north america

    Returns:
    datset: Xarray dataset with 'snow_depth' and 'wet_snow' variables for all Sentinel-1
    image acquistions in area and dates
    """

    # -- preflight checks -- #
    aoi = validate_aoi(aoi)
    dates = validate_dates(*dates)

    assert isinstance(debug, bool), f"Debug keyword must be boolean. Got {debug}"
    assert isinstance(params, list) or isinstance(params, tuple), f"param keyword must be list or tuple. Got {type(params)}"
    assert len(params) == 3, f"List of params must be 3 in order A, B, C. Got {params}"
    A, B, C = params

    assert wet_snow_thresh < 0, f"Wet snow threshold set at {wet_snow_thresh}. This value is positive but should be negative."
    assert freezing_snow_thresh > 0, f"Refreeze threshold set at {freezing_snow_thresh}. This value is negative but should be positive."

    if outfp is not None:
        outfp = Path(outfp).expanduser().resolve()
        assert outfp.parent.exists(), f"Out filepath {outfp}'s directory does not exist"

    # -- Setting up directories and logging -- #
    work_dir = Path(work_dir)
    os.makedirs(work_dir, exist_ok = True)

    setup_logging(log_dir = join(work_dir, 'logs'), debug = debug)
    log = logging.getLogger(__name__)

    # get main stem for automated file naming
    work_stem = f'{pd.to_datetime(dates[0]).date()}_{pd.to_datetime(dates[1]).date()}' if work_dir is not None else work_stem
    
    # -- Downloading S1, FCF, Snowcover -- #

    # start with Sentinel-1 #
    log.info("Downloading sentinel-1 gamma0 backscatter data")

    # get sentinel 1 data search results
    s1_urls = get_sentinel1_urls(start_date = dates[0], stop_date = dates[1], aoi = aoi)
    # Keep only necessary downloads vv, vh, mask.
    s1_urls = [u for u in s1_urls if u.endswith(('_VV.tif', '_VH.tif', '_mask.tif'))]
    # download sentinel urls
    s1_fps = download_urls_parallel(s1_urls, work_dir.joinpath('opera'))

    # generate dataset and start to save s1 data vars. Use zarr to reduce memory load for big arrays
    ds = xr.Dataset()
    ds['vv'] = generate_sentinel1_dataarray(s1_fps, aoi, pol = 'VV', resolution = resolution)

    # grab spatial reference from first time step of VV
    spatial_reference = ds['vv'].isel(time = 0)

    # repeat combining and spatial resampling for VH
    ds['vh'] = generate_sentinel1_dataarray(s1_fps, aoi, pol = 'VH', ref = spatial_reference)
    
    # next VIIRS snowcover #
    log.info("Downloading VIIRS snowcover data")

    # download viirs snow cover fraction for each sentinel 1 overpass date
    snowcover_urls = find_snowcover_urls(start_date = dates[0], stop_date = dates[1], date_list = ds.time.dt.date, aoi = aoi)
    snowcover_fps = download_urls_parallel(snowcover_urls, work_dir.joinpath('snowcover'))

    # generate snow cover dataset
    viirs_snowcover = generate_snowcover_dataarray(snowcover_fps, ref = spatial_reference)

    # resample date times from each day to the exact s1 overpass time. Doubles if two in one day
    viirs_snowcover = convert_snowcover_dates_to_s1_overpasses(viirs_snowcover, ds['vv'])

    # grab out watermask information and snowcover data
    # table 1 https://nsidc.org/sites/default/files/documents/user-guide/multi_vnp10a1f-v002-userguide.pdf
    ds['watermask'] = (viirs_snowcover == 237).median('time')
    # set pixels with over 10% snow cover to snowcovered
    ds['snowcover'] = (viirs_snowcover.where(viirs_snowcover <= 100) > 10).astype(int)
    
    # finally NLCD or Proba-V forest cover fraction #
    log.info("Downloading forest cover fraction data")
    # download fcf and add to dataset ['fcf'] keyword
    # this will us NLCD for within US (very fast) or Proba-v for global (very slow)
    ds['fcf'] = generate_forest_fraction_dataarray(aoi, ref = spatial_reference)

    # -- Preprocessing Sentinel-1 backscatter -- #
    log.info("Preprocessing Sentinel-1 images")

    # water mask dataset
    ds = ims_water_mask(ds)

    # mask out outliers in incidence angle
    # no longer used. Using opera's included shadow and layover mask generate_dataset
    # ds = s1_incidence_angle_masking(ds)
    
    # convert from gamma0 amplitude to dB
    ds = amplitude_to_dB(ds)
    
    # average orbits backscatter to same means
    ds = s1_orbit_averaging(ds)
    # clip outlier values of backscatter to overall mean
    ds = s1_clip_outliers(ds)

    # -- Calulating Snow Index -- #
    log.info("Calculating snow index")
    # calculate delta CR and delta VV
    ds = calc_delta_cross_ratio(ds, A = A)
    ds = calc_delta_vv(ds)

    # calculate delta gamma with delta CR and delta VV with FCF
    ds = calc_delta_gamma(ds, B = B)

    # clip outliers of delta gamma
    ds = clip_delta_gamma_outlier(ds)

    # calculate snow_index from delta_gamma
    ds = calc_snow_index(ds, ims_masking = ims_masking)

    # convert snow index to snow depth
    ds = calc_snow_index_to_snow_depth(ds, C = C)

    ## Wet Snow Flags
    log.info("Flag wet snow")
    # find newly wet snow
    ds = id_newly_wet_snow(ds, wet_thresh = wet_snow_thresh)
    ds = id_wet_negative_si(ds, wet_SI_thresh = wet_SI_thresh)

    # find newly frozen snow
    ds = id_newly_frozen_snow(ds, freeze_thresh = freezing_snow_thresh)

    # make wet_snow flag
    ds = flag_wet_snow(ds)

    ds.attrs['param_A'] = A
    ds.attrs['param_B'] = B
    ds.attrs['param_C'] = C

    ds.attrs['bounds'] = aoi.bounds

    # set dimensions in rioxarray order
    ds = ds.transpose("time", "y", "x")

    if outfp:
        outfp = str(outfp)
        
        ds.to_netcdf(outfp)

    return ds

def retrieval_from_parameters(dataset: xr.Dataset, 
                              A: float, 
                              B: float, 
                              C: float, 
                              wet_SI_thresh: float = 0, 
                              freezing_snow_thresh: float = 2,
                              wet_snow_thresh: float = -2):
    """
    Retrieve snow depth with varied parameter set from an already pre-processed
    dataset.

    Args:
    dataset: Already preprocessed dataset with s1, fcf, ims, deltaVV, merged images, 
    and masking applied.
    A: A parameter
    B: B parameter
    C: C parameter

    Returns:
    dataset: xarray dataset with snow_depth variable calculated from parameters
    """

    # dataset = dataset[['s1','deltaVV','ims','fcf','lidar-sd']]

    # load datast to index
    dataset = dataset.load()

    # calculate delta CR and delta VV
    dataset = calc_delta_cross_ratio(dataset, A = A)

    # calculate delta gamma with delta CR and delta VV with FCF
    dataset = calc_delta_gamma(dataset, B = B)

    # clip outliers of delta gamma
    dataset = clip_delta_gamma_outlier(dataset)

    # calculate snow_index from delta_gamma
    dataset = calc_snow_index(dataset)

    # convert snow index to snow depth
    dataset = calc_snow_index_to_snow_depth(dataset, C = C)

    # find newly wet snow
    dataset = id_newly_wet_snow(dataset, wet_thresh = wet_snow_thresh)
    dataset = id_wet_negative_si(dataset, wet_SI_thresh = wet_SI_thresh)

    # find newly frozen snow
    dataset = id_newly_frozen_snow(dataset, freeze_thresh =  freezing_snow_thresh)

    # make wet_snow flag
    dataset = flag_wet_snow(dataset)

    return dataset
