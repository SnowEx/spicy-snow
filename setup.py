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
REQUIRES_PYTHON = '>=3.9.0'

REQUIRED = [
    'numpy',
    'pandas',
    'xarray',
    'rioxarray',
    'shapely',
    'asf_search',
    'earthaccess',
    'matplotlib',
    'netcdf4',
    'h5py',
    'pygeohydro',
    'zarr'
]

here = os.path.abspath(os.path.dirname(__file__))

# Long description from README
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version="{{VERSION_PLACEHOLDER}}",  # CI replaces this
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
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
