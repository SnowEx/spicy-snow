[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
![version](https://img.shields.io/badge/version-0.0.0-green)

# spicy-snow

<img src="https://github.com/SnowEx/spicy-snow/blob/main/title-img.png" width="800">

Python module to use volumetric scattering at C-band to calculate snow depths from Sentinel-1 imagery using Hans Lieven et al.'s 2021 technique

Lievens et al 2019 - https://www.nature.com/articles/s41467-019-12566-y
- the original nature paper. Methods are at the bottom.

Lievens et al 2021 - https://tc.copernicus.org/articles/16/159/2022/
- Similar process to the 2019 paper, but with wet snow masking and a few other tweaks.

## Proposed Roadmap

- [ ] Design planning
- [ ] Pseudo-code
- [ ] Tests

- [ ] User Inputs: 
    - [ ] dates 
    - [ ] geographic area
    - [ ] others?

- [x] Data products to pull in:
    - [x] Sentinel 1 - orbit file, border noise, thermal noise, radiometric calibration, terrain flattened, gamma_0, range dopper terrain correction, averaged to 100m, mask out incidence angles >70
    - [x] Snow cover (0/1) - Interactive multisensor snow and ice mapping system
    - [ ] Glacier cover from Randolph Glacier Inventory 6.0
    - [x] Forest Cover Fraction from copernicus PROBA-V dataset

- [ ] Processing steps
    - [x] Rescale by mean for all orbits to overall mean
    - [x] Remove observations 3dB above 90th percentile or 3dB below 10th percentile
    - [ ] Calculate \triangle \gamma_{CR} and \triangle \gamma_{VV}
    - [ ] Calculate combined values \gamma
    - [ ] Calculate SI and SD, uses snow cover form IMS too
    - [ ] Set tuning parameters: A = 2, B = 0.5, C = 0.44
    - [ ] Wet snow mask update based on -2dB or +2dB changes
    - [ ] Coarsen to appropriate resolution

- [ ] Output: 
    - [ ] xarray netcdf of snow depths

## Example Installation

```sh
pip install c_snow
```

## Example usage:

```python
from spicy-snow import get_s1_snow_depth, bounding_box
import rioxarray as rxa

# Provide bounding box (user popup or user-provided coordinates)
area = bounding_box() # pop up for user to draw
area = bounding_box(lower_left, upper_left, upper_right, lower_right) # or provide coordinates

# Provide output filename for netcdf
output_netcdf = '/filepath/to/output.netcdf' # should this overwrite file?

# Provide dates as tuple of strings
dates = ("2019-12-01", "2020-04-01")

# Function to actually get data, run processing, returns xarray dataset w/ daily time dimension
s1_sd = get_s1_snow_depth(area, dates, output_netcdf) 
# optional keyword ideas: resolution, overwrite, fitting parameters (A, B, C)

# plot first day of 2020 to check data quality
s1_sd.sel(time = "2020-01-01").plot()
```

## Proposed Directory Structure:

- c-snow
    - download
        - s1_imgs.py
            * s1_img_search(area: shapely geom, dates: pd_time_slice) -> pd dataframe : find dates and url of s1 overpasses and returns granule names
                - https://hyp3-docs.asf.alaska.edu/using/sdk_api/
                - https://nbviewer.org/github/ASFHyP3/hyp3-sdk/blob/main/docs/sdk_example.ipynb
            * download_s1_imgs(urls: pd dataframe) -> xarray : takes granule names of s1 images and download them and return xarray dataset of s1 images
        - ancillary_data.py
            * snow_cover_search(area: shapely geom, dates: (string, string)) -> pd dataframe : find url of IMS snow on/off images
            * download_snow_cover_imgs(urls: pd dataframe, s1_dataset: xarray dataset) -> xarray : takes urls of IMS snow on/off images and download them and adds to the sentinel 1 xarray dataset as a variable/band
            * fcf_search(area: shapely geom, dates: (string, string)) -> pd dataframe : find url of PROBA-V forest cover fraction images
            * download_fcf_imgs(urls: pd dataframe, s1_dataset: xarray dataset) -> xarray : takesPROBA-V forest cover fraction images and download them and adds to the sentinel 1 xarray dataset as a variable/band
    - constants
        - CONSTANTS.py - constants to be used (resolution standards, band/variable names, etc)
    - utils - function to be used in multiple places
        - download_utils.py
        - radar_utils.py
        - c_snow_exceptions.py
    - processing
        - preprocessing.py
            * s1_orbit_averaging(s1_ds: xarray dataset) -> s1_ds : do the mean averaging to allow for different orbits to be compared
            * s1_clip_outliers(s1_ds: xarray dataset) -> s1_ds : clip outliers 3dB above or below 90th percentile
            * calc_d_gamma_cr(s1_ds: xarray dataset) -> s1_ds: calculate d_gamma_cr using s1 images and FCF
            * calc_d_gamma_VV(s1_ds: xarray dataset) -> s1_ds: calculate d_gamma_vv using s1 images and FCF
            * calc_d_gamma(s1_ds: xarray dataset) -> s1_ds: calculate d_gamma using d_gamma_cr and d_gamma_vv
        - snow_index.py
            * calc_snow_index(s1_ds: xarray dataset) -> s1_ds: calculate snow index using d_gamma, wet snow mask and IMS snowon/snowoff
        - wet_snow_mask.py
            * find_wet_snow(s1_ds: xarray dataset) -> s1_ds : find pixels that have either dried (+2dB) or wetted (-2 dB)
            * update_wet_snow(s1_ds: xarray dataset) -> s1_ds : set wet snow mask pixels as wet or dry based on previous time step's classification and threshold change
        - get_s1_snowdepth.py
            * calc_snow_depth(s1_ds: xarray dataset) -> s1_ds : final step using snow index and wet snow mask to calculate snow depths 
    - IO
        - user_input.py
            * get_user_geometry(coordinates: (float, float, float, float)) -> shapely geom : take coordinates from user function call and return shapely geometry w/ error checking
            * get_user_dates(dates: (string, string) -> pd_time_slice : take user input and convert to pandas date/time slice in days w/ error checking
            * get_user_filepath(filepath: string) -> get the filepath to save output at w/ error checking
        - netcdf_export.py
            * save_netcdf(output_fp : string, s1_ds: xarray dataset) -> None : take final s1 dataset and save out variable for retrieved snow depth
- notebooks
    - basic_example.ipynb
- tests
    - test_data
        - expected_output_dataset.pkl
        - expected_input_dataset.pkl
    - test_downloads.py
    - test_inputs.py
    - test_processing.py
    - test_outputs.py
- images
    - cover_img.png
    - example_output.png
- README.md
- environment.yml
- LICENSE
- .gitignore

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

Readme template: https://github.com/othneildrew/Best-README-Template

## Contact

Zach Keskinen: zachary.keskinen@boisestate.edu

Project Link: https://github.com/SnowEx/spicy-snow

## Links to relevant repos
Sentinel 1 Download:
https://github.com/ASFHyP3/hyp3-sdk
https://github.com/ericlindsey/sentinel_query_download

IMS Download:
https://github.com/tylertucker202/tibet_snow_man/blob/master/tutorial/Tibet_snow_man_blog_entry.ipynb
https://github.com/guidocioni/snow_ims

PROBA-V FCF Download:
https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/badge/Contributers-0-green
[contributors-url]: https://github.com/SnowEx/spicy-snow/graphs/contributors

[issues-shield]: https://img.shields.io/badge/Issues-0-yellowgreen
[issues-url]: https://github.com/SnowEx/spicy-snow/issues

[license-shield]: https://img.shields.io/badge/License-MIT-blue
[license-url]: https://github.com/SnowEx/spicy-snow/blob/main/LICENSE
