[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# c-snow
Collaboration to make a python module to use volumetric scattering at C-band to calculate snow depths from Sentinel-1 imagery using Hans Lieven et al.'s 2021 technique

Lievens et al 2019 - https://www.nature.com/articles/s41467-019-12566-y
- the original nature paper. Methods are at the bottom.

Lievens et al 2021 - https://tc.copernicus.org/articles/16/159/2022/
- Similar process to the 2019 paper, but with wet snow masking and a few other tweaks.

## Proposed Roadmap

- [ ] Inputs: 
    - [ ] dates 
    - [ ] geographic area
    - [ ]others?

- [ ] Data products to pull in:
    - [ ] Sentinel 1 - orbit file, border noise, thermal noise, radiometric calibration, terrain flattened, gamma_0, range dopper terrain correction, averaged to 100m, mask out incidence angles >70
    - [ ] Snow cover (0/1) - Interactive multisensor snow and ice mapping system
    - [ ] Glacier cover from Randolph Glacier Inventory 6.0
    - [ ] Forest Cover Fraction from copernicus PROBA-V dataset

- [ ] Processing steps
    - [ ] Rescale by mean for all orbits to overall mean
    - [ ] Remove observations 3dB above 90th percentile or 3dB below 10th percentile
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
from c_snow import get_s1_snow_depth, bounding_box
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

Proposed Directory Structure:

- c-snow
    - s1_imgs.py
    - ancillary_data.py
    - processing.py
    - output.py
- notebooks
    - basic_example.ipynb
- tests
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

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

Readme template: https://github.com/othneildrew/Best-README-Template

## Contact

Project Link: https://github.com/SnowEx/c-snow

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
