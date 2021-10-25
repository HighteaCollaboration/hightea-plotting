#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(
    name='hightea-plotting',
    version='0.1.0',
    license='MIT',
    description='Plotting routines for hightea',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    pymodules='run'
)
