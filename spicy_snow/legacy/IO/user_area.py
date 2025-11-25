"""
Helper function to get bounding box from user
"""

import shapely.geometry
import xarray as xr
import rioxarray as rxa

from typing import Tuple, List

def get_input_area(coords: Tuple[List[str], List[float], bool] = False,
                   img: Tuple[xr.Dataset, xr.DataArray, bool] = False):
    """
    Helper function to return coordinates in bounding box format for retrieval
    Can only provide 1 argument type to this function.
    
    Args:
    coords: list of lat/long coordinates in format [xmin, ymin, xmax, ymax]
    img: xarray dataset or dataArray to extract bounding box from

    Returns:
    area: shapely.geometry.Polygon of bounding box
    """
    args = locals().values()

    assert any(args), "Must provide at least one argument"

    assert len([a for a in args if a != False]) == 1, "Must provide only argument."

    if coords:

        assert len(coords) == 4, f"Provide the 4 coordinates in [xmin, ymin, xmax, ymax] \
format. Provided list {coords} doesn't have 4 elements. Has {len(coords)}."

        if type(coords[0]) == str:
            coords = [float(c) for c in coords]
        
        xmin, ymin, xmax, ymax = coords

        assert xmin < xmax, f"xmin ({xmin}) is greater than xmax ({xmax})"
        assert ymin < ymax, f"ymin ({ymin}) is greater than ymax ({ymax})"

        assert ymin > -90, "Provide in lat/long"
        assert ymax < 90, "Provide in lat/long"

        assert xmin > -180, "Provide in lat/long"
        assert xmax < 180, "Provide in lat/long"

        area = shapely.geometry.box(xmin, ymin, xmax, ymax)

    if img:

        if img.rio.crs != 4326:
            img = img.rio.reproject("EPSG:4326")
        
        area = shapely.geometry.box(*img.rio.bounds())

    return area