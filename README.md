[![MIT License][license-shield]][license-url]
[![PIP](https://img.shields.io/badge/pip-0.1.3-purple)](https://img.shields.io/badge/pip-0.1.3-purple)
[![COVERAGE](https://img.shields.io/badge/coverage-86%25-green)](https://img.shields.io/badge/coverage-86%25-green) 
[![DOI](https://zenodo.org/badge/590243635.svg)](https://zenodo.org/badge/latestdoi/590243635)

# spicy-snow

Python module to use volumetric scattering at C-band to calculate snow depths from Sentinel-1 imagery using Lieven et al.'s 2021 technique.

The relevant papers for this repository technique are:

Lievens et al 2019 - https://www.nature.com/articles/s41467-019-12566-y

Lievens et al 2021 - https://tc.copernicus.org/articles/16/159/2022/

Hoppinen et al 2024 - https://tc.copernicus.org/articles/18/5407/2024/

<img src="https://github.com/SnowEx/spicy-snow/blob/main/title-img.png" width="800">

## Example Installation

```sh
pip install spicy-snow
```

## Example usage:

```python

# Add main repo to path if not pip installed
# import sys
# sys.path.append('path/to/the/spicy-snow/')

from pathlib import Path
from spicy_snow.retrieval import retrieve_snow_depth

# change to your minimum longitude, min lat, max long, max lat
test_aoi = [-114.5, 43, -114, 44]

# can also be a point for point based retrievals
# test_aoi = [-114.2, 43.75]

# you will want to start before the snowfalls since this is a change detection method
dates = ['2020-10-01', '2021-05-01']

# this will be where your results are saved
out_nc = Path('./test.nc').resolve()

# resolution in meters to work at (100, 500, 1km all have been tested).
# default is 100 meters
spatial_resolution = 100

spicy_ds = retrieve_snow_depth(aoi = test_aoi, dates = dates, 
                               work_dir = Path('./spicy-test/').resolve(), 
                               resolution = spatial_resolution,
                               debug=False,
                               outfp=out_nc)
```




## Description of the output netcdf variables.

 - wet_snow: layer showing layers flagged as wet snow (1 = wet, 0 = dry)
 - snow_depth: derived snow depth in meters
 - snowcover: snow coverage binary mask (True = snow on, False = snow-free)
 - fcf: forest coverage percentage
 - vv: gamma0 sentinel-1 VV backscatter in dB
 - vh: gamma0 sentinel-1 VH backscatter in dB

All the other layers are intermediate layers for if you want to explore the processing pipeline.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Coverage instructions

Run the following from the root directory of this project to get a coverage report.

You will need to have the dependencies and `coverage` packages available.

```bash
python -m coverage run -m unittest discover -s ./tests
python -m coverage report
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

Readme template: https://github.com/othneildrew/Best-README-Template

Title image: https://openai.com/dall-e-2/

## Contact

Zach Hoppinen: zmhoppinen@alaska.edu

Project Link: https://github.com/SnowEx/spicy-snow

## Data Sources

Sentinel-1 RTCs are downloaded from the Alaska Satellite Facility and processed by [JPL's Opera Project](https://github.com/opera-adt/RTC). Forest cover within CONUS comes from the National Land Cover Database, otherwise from the Proba-V forest cover. Snow coverage is from the VIIRS snow cover product. See and use citations below. 

## Citations

Work using this repository should cite it and the following datasources
NASA/JPL/OPERA. (2023). OPERA Co-registered Single Look Complex from Sentinel-1 validated product (Version 1) [Data set]. NASA Alaska Satellite Facility Distributed Active Archive Center. https://doi.org/10.5067/SNWG/OPERA_L2_CSLC-S1_V1 Date Accessed: 2025-11-18

Riggs, G. A., Hall, D. K. & Román, M. O. (2019). VIIRS/NPP Snow Cover Daily L3 Global 375m SIN Grid. (VNP10A1, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/VIIRS/VNP10A1.001. [describe subset used if applicable]. Date Accessed 11-18-2025.

U.S. Geological Survey (USGS), 2024, Annual NLCD Collection 1 Science Products (ver. 1.1, June 2025): U.S. Geological Survey data release, https://doi.org/10.5066/P94UXNTS.

## Links to relevant repos/sites

Opera S1 RTC Products:
https://www.earthdata.nasa.gov/data/catalog/asf-opera-l2-cslc-s1-v1-1
https://asf.alaska.edu/datasets/daac/opera/

VIIRS Snow Fractional Coverage
https://nsidc.org/data/vnp10a1/versions/1

Annual National Land Cover Database
https://www.usgs.gov/centers/eros/science/annual-national-land-cover-database

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
