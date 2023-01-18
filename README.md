# c-snow
Collaboration to make a python module to use volumetric scattering at C-band to calculate snow depths from Sentinel-1 imagery using Hans Lieven et al.'s 2021 technique

Lievens et al 2019 - https://www.nature.com/articles/s41467-019-12566-y
- the original nature paper. Methods are at the bottom.

Lievens et al 2021 - https://tc.copernicus.org/articles/16/159/2022/
- Similar process to the 2019 paper, but with wet snow masking and a few other tweaks.

Inputs: 
- dates 
- geographic area
- others?

Data products to pull in:
- Sentinel 1 - orbit file, border noise, thermal noise, radiometric calibration, terrain flattened, gamma_0, range dopper terrain correction, averaged to 100m, mask out incidence angles >70
- Snow cover (0/1) - Interactive multisensor snow and ice mapping system
- Glacier cover from Randolph Glacier Inventory 6.0
- Forest Cover Fraction from copernicus PROBA-V dataset

Processing steps
- Rescale by mean for all orbits to overall mean
- Remove observations 3dB above 90th percentile or 3dB below 10th percentile
- Calculate \triangle \gamma_{CR} and \triangle \gamma_{VV}
- Calculate combined values \gamma
- Calculate SI and SD, uses snow cover form IMS too
- Set tuning parameters: A = 2, B = 0.5, C = 0.44
- Wet snow mask update based on -2dB or +2dB changes
- Coarsen to appropriate resolution

Output: 
- xarray netcdf of snow depths

Example usage:

```python
from c_snow import get_s1_snow_depth, bounding_box

area = bounding_box() # pop up for user to draw or coordinates can be provided with bounding_box(lower_left, upper_left, upper_right, lower_right)
output_netcdf = '/filepath/to/output.netcdf'
get_s1_snow_depth(area, output_netcdf)
```

