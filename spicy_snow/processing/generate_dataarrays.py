from pathlib import Path
from collections import defaultdict
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import shapely
from shapely.geometry import box, Point, Polygon
from pyproj import Transformer
import h5py
from tqdm.auto import tqdm

# faster reprojection utils
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as Rsmp
from rasterio.transform import rowcol

# multithreading
from concurrent.futures import ThreadPoolExecutor

# forest cover
import pygeohydro as gh

from spicy_snow.utils.raster import combine_close_images, da_to01
from spicy_snow.utils.checks import validate_aoi, within_conus
from spicy_snow.utils.download import download_proba_v
from spicy_snow.utils.geometry import generate_point_transform

import logging
log = logging.getLogger(__name__)

def preallocate_output(times, y, x, dtype, crs, transform, zarr_path = None):

    data = np.zeros(
        (len(times), len(y), len(x)),
        dtype=dtype,
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
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform) # transform

    if zarr_path is not None:
        zarr_path = Path(zarr_path)
        da.to_zarr(zarr_path, mode="w")

    return da
# def generate_sentinel1_dataarray(
#     s1_fps,
#     aoi,
#     pol,
#     zarr_path=None,
#     ref=None,
#     resolution=(100, 100),
#     max_workers=8,
# ):

#     aoi = validate_aoi(aoi)
#     pol = pol.lower()

#     # Filter relevant products
#     pol_fps = [f for f in s1_fps if f.stem.lower().endswith(pol)]

#     # Extract timestamps + tracks
#     times = []
#     tracks = []
#     for fp in pol_fps:
#         t = pd.to_datetime(fp.stem.split('_')[4], format='%Y%m%dT%H%M%SZ')
#         times.append(t)

#         track = int(fp.stem.split('_')[3].split('-')[0][1:])
#         tracks.append(track)

#     # ---- Build reference grid ----
#     if ref is None:
#         ref = xr.open_dataarray(pol_fps[0], chunks="auto")[0]
#         assert ref.rio.crs.is_projected
#         ref = ref.rio.reproject(dst_crs=ref.rio.crs, resolution=resolution)
#         ref = ref.rio.reproject("EPSG:4326")
#         if isinstance(aoi, box):
#             ref = ref.rio.clip_box(*aoi.bounds).rio.pad_box(*aoi.bounds)
#         elif isinstance(aoi, Point):
#             ref = ref.sel(x = aoi.x, y = aoi.y, method = 'nearest')

#     dst_crs = ref.rio.crs
#     dst_transform = ref.rio.transform()
#     dst_shape = ref.shape  # (y, x)

#     # ---- Preallocate Zarr-backed DataArray ----
#     zarr = preallocate_output(ref, times, zarr_path=zarr_path)
#     zarr = zarr.assign_coords(track=("time", tracks))

#     # Direct access to Zarr array (no .loc)
#     z = zarr.data

#     def process_one(args):
#         fp, idx = args

#         with rasterio.open(fp) as src:
#             img = src.read(1).astype("float32")

#             mask_fp = fp.parent / f"{fp.stem[:-3]}_mask.tif"
#             if mask_fp.exists():
#                 with rasterio.open(mask_fp) as m:
#                     mask = (m.read(1) == 0)
#                 img = np.where(mask, img, np.nan)

#             # initialize with NaNs, not zeros
#             dst = np.full(dst_shape, np.nan, dtype="float32")

#             reproject(
#                 img,
#                 dst,
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=dst_transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.bilinear,
#                 src_nodata=0,
#                 dst_nodata=np.nan,
#             )

#         return idx, dst

#     # ---- Parallel projection + writing ----
#     with ThreadPoolExecutor(max_workers=max_workers) as ex:
#         for idx, dst in tqdm(
#             ex.map(process_one, [(fp, i) for i, fp in enumerate(pol_fps)]),
#             total=len(pol_fps),
#             desc=f"Reprojecting + inserting {pol}",
#         ):
#             z[idx, :, :] = dst  # FAST direct Zarr write

#     # Optionally collapse near-duplicate timesteps
#     out = combine_close_images(zarr.sortby("time"))

#     if zarr_path is not None:
#         out = xr.open_zarr(zarr_path)

#     return out

def da_to_2d_array(da, y, x, crs, transform):
    da = da.sel(x = x, y = y, method = 'nearest')
    da = da.data.reshape(1, 1)
    da = xr.DataArray(da, dims = ['y', 'x'], coords = {'y': [y], 'x': [x]})
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    return da

def sample_resampled_average(src, img, aoi, target_res):
    """
    Sample the VALUE of a resampled grid (with target resolution) at a point
    without computing the whole resampled raster.

    Parameters
    ----------
    src : rasterio dataset
        Source raster (with src.transform & src.crs)
    img : np.ndarray
        2D array of original values
    aoi : shapely Point (EPSG:4326)
        Point to sample
    target_res : tuple(float, float)
        (xres_target, yres_target) for resampled grid
    """

    # ---- 1. Convert AOI: WGS84 -> raster CRS ----
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(aoi.x, aoi.y)

    # ---- 2. Original raster resolution (may be negative) ----
    xres_src, yres_src = src.res
    yres_src = abs(yres_src)

    xres_tgt, yres_tgt = target_res
    yres_tgt = abs(yres_tgt)

    # ---- 3. Scaling factor (# of original pixels per target pixel) ----
    scale_x = xres_tgt / xres_src
    scale_y = yres_tgt / yres_src

    # ---- 4. Original pixel indices ----
    row_src, col_src = rasterio.transform.rowcol(src.transform, x, y)

    # ---- 5. Resampled pixel index ----
    row_tgt = int(row_src / scale_y)
    col_tgt = int(col_src / scale_x)

    # ---- 6. Original window that corresponds to this target cell ----
    r0 = int(row_tgt * scale_y)
    r1 = int((row_tgt + 1) * scale_y)

    c0 = int(col_tgt * scale_x)
    c1 = int((col_tgt + 1) * scale_x)

    # Clip to bounds
    r0 = max(0, r0); r1 = min(img.shape[0], r1)
    c0 = max(0, c0); c1 = min(img.shape[1], c1)

    # ---- 7. Compute average of the resampled cell ----
    window = img[r0:r1, c0:c1]
    return np.nanmean(window)

def bilinear_resampled_value(src, img, aoi, target_res):
    """
    Bilinear interpolation at AOI as if the raster were resampled 
    to target_res via rasterio.reproject(..., bilinear).
    """
    # --- CRS transform ---
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(aoi.x, aoi.y)

    # --- Original pixel size ---
    xres_src, yres_src = src.res
    yres_src = abs(yres_src)

    # --- Target pixel size ---
    xres_tgt, yres_tgt = target_res
    yres_tgt = abs(yres_tgt)

    # --- Fractions in the target grid ---
    row_src, col_src = rasterio.transform.rowcol(src.transform, x, y, op=float)
    row_tgt = row_src * (yres_src / yres_tgt)
    col_tgt = col_src * (xres_src / xres_tgt)

    # --- Back-map to fractional original indices ---
    row_f = row_tgt * (yres_tgt / yres_src)
    col_f = col_tgt * (xres_tgt / xres_src)

    r0 = int(np.floor(row_f))
    c0 = int(np.floor(col_f))
    r1 = r0 + 1
    c1 = c0 + 1

    # bounds check
    if r0 < 0 or c0 < 0 or r1 >= img.shape[0] or c1 >= img.shape[1]:
        return np.nan

    # --- 4 neighbors from the original raster ---
    Q11 = img[r0, c0]
    Q21 = img[r0, c1]
    Q12 = img[r1, c0]
    Q22 = img[r1, c1]

    dr = row_f - r0
    dc = col_f - c0

    # --- Bilinear interpolation ---
    top = Q11 * (1-dc) + Q21 * dc
    bottom = Q12 * (1-dc) + Q22 * dc
    return top * (1-dr) + bottom * dr


def generate_sentinel1_dataarray(
    s1_fps,
    aoi,
    pol,
    zarr_path=None,
    ref=None,
    resolution=(100, 100),
    max_workers = 8
):
    if isinstance(resolution, float) or isinstance(resolution, int):
        resolution = (resolution, resolution)

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
        if isinstance(aoi, shapely.geometry.point.Point):
            crs = ref.rio.crs
            xres, yres = ref.rio.resolution()
            transform = generate_point_transform(x = aoi.x, y = aoi.y, xres = xres, yres = yres)
            ref = da_to_2d_array(ref, aoi.y, aoi.x, crs, transform)
        else:
            ref = ref.rio.clip_box(*aoi.bounds).rio.pad_box(*aoi.bounds)

    dst_crs = ref.rio.crs
    dst_transform = ref.rio.transform()
    dst_shape = ref.shape

    # ---- Preallocate Zarr-backed DataArray ----
    zarr = preallocate_output(times, 
                              ref.y, 
                              ref.x,
                              dtype = ref.dtype,
                              crs = ref.rio.crs,
                              transform= ref.rio.transform(),
                              zarr_path=zarr_path)
    
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

            if isinstance(aoi, shapely.geometry.point.Point):
                value = sample_resampled_average(src, img, aoi, resolution)
                dst = np.array([[value]], dtype=img.dtype)  # shape (1,1)
                return idx, dst

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
    if isinstance(aoi, Point):
        aoi = aoi.buffer(0.001)
    g = gpd.GeoSeries([box(*aoi.bounds)], crs='EPSG:4326')
    fcf_da = gh.nlcd_bygeom(geometry = g)[0]['canopy_2021']

    if isinstance(aoi, Point):
        fcf_da = da_to_2d_array(fcf_da, aoi.y, aoi.x, fcf_da.crs, fcf_da.rio.transform())

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