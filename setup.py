#!/usr/bin/env python
import os
import sys

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name='crispy',
    version="0.9", 
    author='Maxime Rizzo',
    author_email = 'maxime.j.rizzo@nasa.gov',
    url = 'https://github.com/mjrfringes/crispy',
    packages =['crispy','crispy.tools'],
    license = ['GNU GPLv3'],
    description ='The Coronagraph and Rapid Imaging Spectrograph in Python',
    package_dir = {"crispy":'crispy', "crispy.tools":'crispy/tools'},
    include_package_data=True,
    classifiers = [
        'Development Status :: 3 - Alpha',#5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
        ],
    include_dirs = ['crispy','crispy/tools'],
    install_requires = ['numpy','scipy','matplotlib','astropy','photutils'],
)
