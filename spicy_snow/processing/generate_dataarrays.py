from pathlib import Path
from collections import defaultdict
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from shapely.geometry import box
import h5py
from tqdm.auto import tqdm

# faster reprojection utils
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as Rsmp

# multithreading
from concurrent.futures import ThreadPoolExecutor

# forest cover
import pygeohydro as gh

import sys
sys.path.append('/Users/zmhoppinen/Documents/spicy-snow/spicy_snow/utils')
from raster import combine_close_images, da_to01
from checks import validate_aoi, within_conus
from download import download_proba_v

import logging
log = logging.getLogger(__name__)

def preallocate_output(ref_da, times, zarr_path = None):
    y = ref_da["y"]
    x = ref_da["x"]

    data = np.zeros(
        (len(times), len(y), len(x)),
        dtype=ref_da.dtype,
    )

    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": y,
            "x": x,
        }
    )
    
    # save spatial reference information
    da = da.rio.write_crs(ref_da.rio.crs)
    da = da.rio.write_transform(ref_da.rio.transform())

    if zarr_path is not None:
        zarr_path = Path(zarr_path)
        da.to_zarr(zarr_path, mode="w")

    return da

def generate_sentinel1_dataarray(
    s1_fps,
    aoi,
    pol,
    zarr_path=None,
    ref=None,
    resolution=(100, 100),
    max_workers=8,
):

    aoi = validate_aoi(aoi)
    pol = pol.lower()

    # Filter relevant products
    pol_fps = [f for f in s1_fps if f.stem.lower().endswith(pol)]

    # Extract timestamps + tracks
    times = []
    tracks = []
    for fp in pol_fps:
        t = pd.to_datetime(fp.stem.split('_')[4], format='%Y%m%dT%H%M%SZ')
        times.append(t)

        track = int(fp.stem.split('_')[3].split('-')[0][1:])
        tracks.append(track)

    # ---- Build reference grid ----
    if ref is None:
        ref = xr.open_dataarray(pol_fps[0], chunks="auto")[0]
        assert ref.rio.crs.is_projected
        ref = ref.rio.reproject(dst_crs=ref.rio.crs, resolution=resolution)
        ref = ref.rio.reproject("EPSG:4326")
        ref = ref.rio.clip_box(*aoi.bounds).rio.pad_box(*aoi.bounds)

    dst_crs = ref.rio.crs
    dst_transform = ref.rio.transform()
    dst_shape = ref.shape  # (y, x)

    # ---- Preallocate Zarr-backed DataArray ----
    zarr = preallocate_output(ref, times, zarr_path=zarr_path)
    zarr = zarr.assign_coords(track=("time", tracks))

    # Direct access to Zarr array (no .loc)
    z = zarr.data

    def process_one(args):
        fp, idx = args

        with rasterio.open(fp) as src:
            img = src.read(1).astype("float32")

            mask_fp = fp.parent / f"{fp.stem[:-3]}_mask.tif"
            if mask_fp.exists():
                with rasterio.open(mask_fp) as m:
                    mask = (m.read(1) == 0)
                img = np.where(mask, img, np.nan)

            # initialize with NaNs, not zeros
            dst = np.full(dst_shape, np.nan, dtype="float32")

            reproject(
                img,
                dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=0,
                dst_nodata=np.nan,
            )

        return idx, dst

    # ---- Parallel projection + writing ----
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, dst in tqdm(
            ex.map(process_one, [(fp, i) for i, fp in enumerate(pol_fps)]),
            total=len(pol_fps),
            desc=f"Reprojecting + inserting {pol}",
        ):
            z[idx, :, :] = dst  # FAST direct Zarr write

    # Optionally collapse near-duplicate timesteps
    out = combine_close_images(zarr.sortby("time"))

    if zarr_path is not None:
        out = xr.open_zarr(zarr_path)

    return out

def generate_sentinel1_dataarray(
    s1_fps,
    aoi,
    pol,
    zarr_path=None,
    ref=None,
    resolution=(100, 100),
    max_workers=8,
):

    aoi = validate_aoi(aoi)
    pol = pol.lower()

    # Filter relevant products
    pol_fps = [f for f in s1_fps if f.stem.lower().endswith(pol)]

    # Extract timestamps + tracks
    times = []
    tracks = []
    for fp in pol_fps:
        t = pd.to_datetime(fp.stem.split('_')[4], format='%Y%m%dT%H%M%SZ')
        times.append(t)

        track = int(fp.stem.split('_')[3].split('-')[0][1:])
        tracks.append(track)

    # ---- Build reference grid ----
    if ref is None:
        ref = xr.open_dataarray(pol_fps[0], chunks="auto")[0]
        assert ref.rio.crs.is_projected
        ref = ref.rio.reproject(dst_crs=ref.rio.crs, resolution=resolution)
        ref = ref.rio.reproject("EPSG:4326")
        ref = ref.rio.clip_box(*aoi.bounds).rio.pad_box(*aoi.bounds)

    dst_crs = ref.rio.crs
    dst_transform = ref.rio.transform()
    dst_shape = ref.shape  # (y, x)

    # ---- Preallocate Zarr-backed DataArray ----
    zarr = preallocate_output(ref, times, zarr_path=zarr_path)
    zarr = zarr.assign_coords(track=("time", tracks))

    # Direct access to Zarr array (no .loc)
    z = zarr.data

    def process_one(args):
        fp, idx = args

        with rasterio.open(fp) as src:
            img = src.read(1).astype("float32")

            mask_fp = fp.parent / f"{fp.stem[:-3]}_mask.tif"
            if mask_fp.exists():
                with rasterio.open(mask_fp) as m:
                    mask = (m.read(1) == 0)
                img = np.where(mask, img, np.nan)

            # initialize with NaNs, not zeros
            dst = np.full(dst_shape, np.nan, dtype="float32")

            reproject(
                img,
                dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=0,
                dst_nodata=np.nan,
            )

        return idx, dst

    # ---- Parallel projection + writing ----
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, dst in tqdm(
            ex.map(process_one, [(fp, i) for i, fp in enumerate(pol_fps)]),
            total=len(pol_fps),
            desc=f"Reprojecting + inserting {pol}",
        ):
            z[idx, :, :] = dst  # FAST direct Zarr write

    # Optionally collapse near-duplicate timesteps
    out = combine_close_images(zarr.sortby("time"))

    if zarr_path is not None:
        out = xr.open_zarr(zarr_path)

    return out

def generate_snowcover_dataarray(snowcover_fps, ref = None):
    # Organize tiles by date
    tiles_by_date = defaultdict(list)

    for fp in snowcover_fps:
        with h5py.File(fp, 'r') as f:
            # grab datetime from properties
            # based on this documentation. Section 1.2.3
            # https://nsidc.org/sites/default/files/documents/user-guide/multi_vnp10a1f-v002-userguide.pdf
            dt = pd.to_datetime(fp.stem.split('.')[1], format = 'A%Y%j')
            
            # get snow cover data
            data = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['Data Fields']['CGF_NDSI_Snow_Cover'][:]

            # get x and y dimensions
            xdim = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['XDim'][:]
            ydim = f['HDFEOS']['GRIDS']['VIIRS_Grid_IMG_2D']['YDim'][:]
            
            # generate xarray dataarray
            da_tile = xr.DataArray(
                data,
                coords={'y': ydim, 'x': xdim},
                dims=['y', 'x'],
            )
            
            tiles_by_date[dt].append(da_tile)

    # For each date, combine tiles spatially, then stack along time
    daily_arrays = []
    for dt, tile_list in tiles_by_date.items():
        # mosaic tiles for this day
        da_day = xr.combine_by_coords(tile_list, combine_attrs='override')
        # add time dimension
        da_day = da_day.expand_dims(time=[dt])
        daily_arrays.append(da_day)

    # concatenate all days along time
    snowcover_da = xr.concat(daily_arrays, dim='time').sortby('time')
    snowcover_da.name = 'snowcover'

    # convert from NASA's sinusoidal project
    # proj string comes from Table 3 of
    # https://nsidc.org/sites/default/files/documents/user-guide/multi_vnp10a1f-v002-userguide.pdf
    src_crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    snowcover_da = snowcover_da.rio.write_crs(src_crs)

    if ref is not None:
        snowcover_da = snowcover_da.rio.reproject_match(ref)
    
    return snowcover_da

def convert_snowcover_dates_to_s1_overpasses(viirs_snowcover, s1_da):
    precise_snowcover = xr.zeros_like(s1_da)
    for s1_time in s1_da.time:
        mask = viirs_snowcover.time.dt.date == s1_time.dt.date
        if len(viirs_snowcover.sel(time=mask)) == 0:
            raise ValueError(f"No VIIRS snowcover found for S1 date {s1_time.values}")

        precise_snowcover.loc[dict(time = s1_time)] = viirs_snowcover.sel(time = mask).isel(time = 0)
    return precise_snowcover

def get_nlcd(aoi):
    g = gpd.GeoSeries([box(*aoi.bounds)], crs='EPSG:4326')
    fcf_da = gh.nlcd_bygeom(geometry = g)[0]['canopy_2021']

    return fcf_da

def generate_forest_fraction_dataarray(aoi, ref = None) -> xr.Dataset:
    """
    Download PROBA-V forest-cover-fraction images.

    Args:
    aoi: 4 element box of AOI

    Returns:
    dataset: Forest cover fraction dataarray over aoi. NLCD if in US otherwise Proba-v
    """
    aoi = validate_aoi(aoi)
    # first check if in us
    if not within_conus(aoi): 
        log.info(f'AOI outside of CONUS. Using Proba-V datasets')
        tmp_dir = tempfile.gettempdir()
        fcf = xr.open_dataarray(download_proba_v(), tmp_dir.joinpath('fcf.tif'))[0]
    else: 
        log.info(f'AOI inside of CONUS. NLCD 2021 Forest Cover')
        fcf = get_nlcd(aoi)

    # reproject FCF and clip to match dataset
    if ref is not None:
        log.debug(f"Clipping FCF to {ref.rio.bounds()}")
        fcf = fcf.rio.reproject_match(ref)

    # if max is greater than 1 () set to 0-1
    if fcf.max() > 1:
        log.debug("fcf max > 1 so normalizing from 0 to 100 -> 0 to 1")
        fcf = da_to01(fcf, old_min = 0, old_max = 100)
        log.debug(f"New fcf max is {fcf.max()} and min is {fcf.min()}")
    
    assert fcf.max() <= 1, "Forest cover fraction must be bounded 0-1"
    assert fcf.min() >= 0, "Forest cover fraction must be bounded 0-1"

    log.debug(f'FCF min: {fcf.min()}')
    log.debug(f'FCF max: {fcf.max()}')
    log.debug(f'FCF mean: {fcf.mean()}')

    fcf.name = 'fcf'

    return fcf

# End of file