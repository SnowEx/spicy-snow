#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import find_packages, setup

NAME = 'spicy-snow'
DESCRIPTION = 'Snow Depth Retrievals from Sentinel-1 Backscatter.'
URL = 'https://github.com/SnowEx/spicy-snow'
EMAIL = 'zachhoppinen@gmail.com'
AUTHOR = 'Zach Hoppinen'
REQUIRES_PYTHON = '>=3.9,<3.12'

REQUIRED = [
    'numpy',
    'pandas',
    'xarray',
    'geopandas',
    'rioxarray',
    'shapely',
    'asf_search',
    'earthaccess',
    'matplotlib',
    'netcdf4',
    'h5py',
    'pygeohydro',
    'zarr',
    'dask'
]

here = os.path.abspath(os.path.dirname(__file__))

# Long description from README
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

import os

VERSION = os.getenv("SPICY_SNOW_VERSION", "0.0.0.dev0")

setup(
    name=NAME,
    version=VERSION,   # pass the resolved version here
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests/*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: Implementation :: CPython',
    ],
)
