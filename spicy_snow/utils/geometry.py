from shapely.geometry import box, Point
from rasterio import Affine

def point_to_box(aoi, xres, yres):
    # Use one pixel size around the point
    # xres, yres = ref.rio.resolution()  # current pixel size
    return box(aoi.x - xres/2, aoi.y - yres/2, aoi.x + xres/2, aoi.y + yres/2)

def generate_point_transform(y, x, xres, yres):
    # Optional: assign a minimal affine transform (pixel centered on the point)
    
    xres, yres = 1, 1  # dummy, or use original pixel size if available
    transform = Affine.translation(x - xres/2, y - yres/2) * Affine.scale(xres, yres)
    return transform
    # ref_point_da = ref_point_da.rio.write_transform(transform)
