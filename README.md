[![MIT License][license-shield]][license-url]
[![PIP](https://img.shields.io/badge/pip-0.1.2-purple)](https://img.shields.io/badge/pip-0.1.2-purple)
[![COVERAGE](https://img.shields.io/badge/coverage-86%25-green)](https://img.shields.io/badge/coverage-86%25-green) 
[![DOI](https://zenodo.org/badge/590243635.svg)](https://zenodo.org/badge/latestdoi/590243635)

# spicy-snow

Python module to use volumetric scattering at C-band to calculate snow depths from Sentinel-1 imagery using Lieven et al.'s 2021 technique.

The relevant papers for this repository technique are:

Lievens et al 2019 - https://www.nature.com/articles/s41467-019-12566-y

Lievens et al 2021 - https://tc.copernicus.org/articles/16/159/2022/

<img src="https://github.com/SnowEx/spicy-snow/blob/main/title-img.png" width="800">

## Example Installation

```sh
pip install spicy-snow
```

## Example usage:

```python
from pathlib import Path

# Add main repo to path if you haven't added with conda-develop
# import sys
# sys.path.append('path/to/the/spicy-snow/')

from spicy_snow.retrieval import retrieve_snow_depth
from spicy_snow.IO.user_dates import get_input_dates

# change to your minimum longitude, min lat, max long, max lat
area = shapely.geometry.box(-113.2, 43, -113, 43.4)

# this will be where your results are saved
out_nc = Path('~/Desktop/spicy-test/test.nc').expanduser()

# this will generate a tuple of dates from the previous August 1st to this date
dates = get_input_dates('2021-04-01') # run on all s1 images from (2020-08-01, 2021-04-01) in this example

spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                               work_dir = Path('~/Desktop/spicy-test/').expanduser(), 
                               job_name = f'testing_spicy',
                               existing_job_name = 'testing_spicy',
                               debug=False,
                               outfp=out_nc)
```

### Running over large areas/memory issues

If you are running out of memory or running over multiple degrees of latitude this code snippet should get you started on batch processing swathes.

```python
from shapely import geometry
from itertools import product
for lon_min, lat_min in product(range(-117, -113), range(43, 46)):
    area = shapely.geometry.box(lon_min, lat_min, lon_min + 1, lat_min + 1)
    out_nc = Path(f'~/Desktop/spicy-test/swath_{lon_min}-{lon_min + 1}_{lat_min}-{lat_min + 1}.nc').expanduser()
    if out_nc.exists():
        continue

    spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                                work_dir = Path('~/scratch/spicy-lowman-quadrant/data/').expanduser(), 
                                job_name = f'spicy-lowman-{lon_min}-{lon_min + 1}_{lat_min}-{lat_min + 1}', # v1
                                existing_job_name = f'spicy-lowman-{lon_min}-{lon_min + 1}_{lat_min}-{lat_min + 1}', # v1
                                debug=False,
                                outfp=out_nc)

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

Zach Keskinen: zacharykeskinen@boisestate.edu

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
