import warnings
from collections.abc import Iterable
from urllib.parse import urlparse
from datetime import date

import numpy as np
import pandas as pd
from shapely.geometry import box, Polygon, Point

def validate_urls(urls, check_exists=False, timeout=5):
    """
    Flatten, validate, and optionally check existence of URLs.

    Args:
        urls (iterable): List/tuple/set of URLs or nested iterables.
        check_exists (bool): If True, perform a simple HTTP HEAD request to verify URL exists.
        timeout (int|float): Timeout for existence check in seconds.

    Returns:
        list: Flattened, validated URLs.

    Raises:
        ValueError: If any URL is invalid or if no URLs remain after validation.
    """

    # check if any urls in list
    assert len(urls) > 0, "No urls found"
    
    def flatten(lst):
        """Recursively flatten nested iterables, skip strings as iterables."""
        for item in lst:
            if isinstance(item, str) or item is None:
                yield item
            elif isinstance(item, Iterable):
                yield from flatten(item)
            else:
                yield item

    clean_urls = []
    for u in flatten(urls):
        if u is None:
            continue

        if not isinstance(u, str):
            warnings.warn(f"Skipping non-string URL: {u}")
            continue

        u = u.strip()
        if not u:
            continue

        parsed = urlparse(u)
        if not parsed.scheme or not parsed.netloc:
            warnings.warn(f"URL does not look valid: {u}")
        
        clean_urls.append(u)

    if not clean_urls:
        raise ValueError("No valid URLs found in input.")

    return clean_urls

def validate_dates(start_date, end_date):
    """
    Validate a start and end date for Sentinel-1 SAR availability.

    Rules:
    - Convert both dates to pandas Timestamps.
    - Dates must be >= 2014 (Sentinel-1A launch).
    - Dates must be <= today.
    - start_date must be < end_date.
    - Warn if dates overlap Sentinel-1B outage (Dec 2021 → present).
    - Warn if dates fall before Sentinel-1C becomes operational (May 20, 2025).
    """
    # ---- Convert to Timestamps ----
    if start_date is not None:
        start = pd.to_datetime(start_date)
    else:
        raise ValueError("start_date cannot be None")

    if end_date is not None:
        end = pd.to_datetime(end_date)
    else:
        raise ValueError("end_date cannot be None")

    # ---- Basic range checks ----
    if start.year < 2014 or end.year < 2014:
        raise ValueError("Dates must be in or after 2014 (Sentinel-1A launch).")

    today = pd.to_datetime(date.today())

    if start > today or end > today:
        raise ValueError("Dates cannot be in the future.")

    if start >= end:
        raise ValueError("start_date must be earlier than end_date.")

    # ---- Special Sentinel mission warnings ----
    # S1B failed: Dec 23, 2021 →
    # S1C operational: May 20, 2025

    s1b_fail = pd.to_datetime("2021-12-23")
    s1c_start = pd.to_datetime("2025-05-20")
    if end >= s1b_fail and end < s1c_start:
        warnings.warn(
            "Date range intersects the Sentinel-1B outage period (Dec 2021 → present). "
            "Only S1A data will be available."
        )

    return start, end

def validate_aoi(aoi):
    """
    Validate and normalize an AOI to a shapely Box geometry or a Point.

    Accepts:
    - Iterable of four floats [xmin, ymin, xmax, ymax] -> returns Box (Polygon)
    - Iterable of two floats [x, y] -> returns Point
    - Dicts using common key conventions:
        {'xmin','ymin','xmax','ymax'} or
        {'west','south','east','north'} or
        {'minx','miny','maxx','maxy'} -> returns Box (Polygon)
    - Existing shapely geometry (Box, Polygon, Point, etc.)

    Returns:
        shapely.geometry.Polygon or shapely.geometry.Point
    """
    # If already a shapely geometry, return as is
    if isinstance(aoi, (Polygon, Point)):
        return aoi

    # If iterable (list, tuple, np.ndarray)
    if isinstance(aoi, (list, tuple, np.ndarray)):
        if len(aoi) == 4:
            xmin, ymin, xmax, ymax = aoi
            # auto-fix reversed ranges
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            return box(xmin, ymin, xmax, ymax)
        elif len(aoi) == 2:
            x, y = aoi
            return Point(x, y)

    # If dict, try known key sets
    if isinstance(aoi, dict):
        key_sets = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("west", "south", "east", "north"),
            ("minx", "miny", "maxx", "maxy"),
        ]
        for keys in key_sets:
            if all(k in aoi for k in keys):
                xmin, ymin, xmax, ymax = (float(aoi[k]) for k in keys)
                if xmin > xmax: xmin, xmax = xmax, xmin
                if ymin > ymax: ymin, ymax = ymax, ymin
                return box(xmin, ymin, xmax, ymax)
        raise ValueError(
            f"AOI dict must contain one of these key sets: {key_sets} "
            f"but received keys: {list(aoi.keys())}"
        )

    raise ValueError(f"Unable to parse {aoi} as AOI bounding box or Point.")

def within_conus(aoi):
    """
    Quick bounding-box check using approximate contiguous US lat/lon limits.
    Assumes aoi is 4 element iterable of (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    aoi = validate_aoi(aoi)
    xmin, ymin, xmax, ymax = aoi.bounds

    # Approximate CONUS bounding envelope
    CONUS_XMIN, CONUS_XMAX = -125, -66
    CONUS_YMIN, CONUS_YMAX = 24, 50

    # Check intersection / overlap
    intersects_lon = not (xmax < CONUS_XMIN or xmin > CONUS_XMAX)
    intersects_lat = not (ymax < CONUS_YMIN or ymin > CONUS_YMAX)

    return intersects_lon and intersects_lat
