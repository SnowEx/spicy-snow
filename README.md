[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
![version](https://img.shields.io/badge/version-0.0.0-green)

# spicy-snow

Python module to use volumetric scattering at C-band to calculate snow depths from Sentinel-1 imagery using Lieven et al.'s 2021 technique.

The relevant papers for this repository technique are:

Lievens et al 2019 - https://www.nature.com/articles/s41467-019-12566-y

Lievens et al 2021 - https://tc.copernicus.org/articles/16/159/2022/

<img src="https://github.com/SnowEx/spicy-snow/blob/main/title-img.png" width="800">

## Proposed Roadmap

- [x] Design planning
- [x] Pseudo-code
- [x] Logging system

- [x] User Inputs: 
    - [x] dates 
    - [x] geographic area

- [x] Data products to pull in:
    - [x] Sentinel 1 - orbit file, border noise, thermal noise, radiometric calibration, terrain flattened, gamma_0, range dopper terrain correction, averaged to 100m, mask out incidence angles >70
    - [x] Snow cover (0/1) - Interactive multisensor snow and ice mapping system
    - [ ] Glacier cover from Randolph Glacier Inventory 6.0
    - [x] Forest Cover Fraction from copernicus PROBA-V dataset
    - [ ] Water cover from PROBA-V dataset

- [x] Processing steps
    - [x] Rescale by mean for all orbits to overall mean
    - [x] Remove observations 3dB above 90th percentile or 3dB below 10th percentile
    - [x] Calculate \triangle \gamma_{CR} and \triangle \gamma_{VV}
    - [x] Calculate combined values \gamma
    - [x] Calculate previous SI using weighted +- 5/11 days SI
    - [x] Calculate SI and SD, uses snow cover from IMS too
    - [x] Wet snow mask update based on -2dB or +2dB changes, negative SI
    - [x] Coarsen to appropriate resolution

- [x] Output: 
    - [x] xarray netcdf of snow depths
    
- [x] Tests

## Example Installation

```sh
pip install c_snow
```

## Example usage:

```python
from spicy_snow import retrieve_snow_depth
from spicy_snow.IO.user_dates import get_input_dates

import shapely

# Provide bounding box (EPSG:4326 user-provided coordinates)
area = shapely.geometry.box(-115, 43, -114, 44)

# Get tuple of dates. Provided date is ending date and start date is always prior August 1st
dates = get_input_dates("2020-04-01")

# Function to actually get data, run processing, returns xarray dataset w/ daily time dimension
s1_sd = get_s1_snow_depth(area, dates, work_dir = './idaho_retrieval/) 

# work_dir will be created if not present 
# optional keyword ideas: job_name, fitting parameters (A, B, C), exisiting_job_name, outfp
# `outfp = './idaho_ret.nc` will output datset to netcdf

# plot first day of 2020 to check data quality
s1_sd.sel(time = "2020-01-01").plot()

# save as pickle file
# dump completed dataset to data directory
with open('./idaho_retrieval/spicy_test.pkl', 'wb') as f:
    pickle.dump(ds, f)
```

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

Title image: https://openai.com/dall-e-2/

## Contact

Zach Keskinen: zachary.keskinen@boisestate.edu

Project Link: https://github.com/SnowEx/spicy-snow

## Links to relevant repos/sites

Sentinel 1 Download:
https://github.com/ASFHyP3/hyp3-sdk
https://github.com/asfadmin/Discovery-asf_search

IMS Download:
https://github.com/tylertucker202/tibet_snow_man/blob/master/tutorial/Tibet_snow_man_blog_entry.ipynb
https://github.com/guidocioni/snow_ims

PROBA-V FCF Download:
https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif

Xarray:
https://github.com/pydata/xarray

Rioxarray:
https://github.com/corteva/rioxarray

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/badge/Contributers-0-green
[contributors-url]: https://github.com/SnowEx/spicy-snow/graphs/contributors

[issues-shield]: https://img.shields.io/badge/Issues-0-yellowgreen
[issues-url]: https://github.com/SnowEx/spicy-snow/issues

[license-shield]: https://img.shields.io/badge/License-MIT-blue
[license-url]: https://github.com/SnowEx/spicy-snow/blob/main/LICENSE
