#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2018 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of nsfds2
#
# nsfds2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# nsfds2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with nsfds2. If not, see <http://www.gnu.org/licenses/>.
#
#
# Creation Date : mar. 10 avril 2018 17:52:42 CEST
# Last Modified : ven. 11 mai 2018 16:13:55 CEST
"""
-----------

setup file for nsfds2

-----------
"""

import platform
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        """ https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py """
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


if platform.system() == 'Windows':
#    libraries = ['msvcrt']
    libraries = []
    extra_compile_args = ["-O2"]
    extra_link_args = []
else:
    libraries = ['m']
    extra_compile_args = ["-O2", "-fopenmp"]
#    extra_compile_args = ["-Ofast", "-fopenmp"]
    extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        'nsfds3.cpgrid.cutils',
        ["nsfds3/cpgrid/cutils.c"],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='nsfds3',
    description="Finite difference solver for Navier-Stokes equations",
    #    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    version="0.2.0",
    license="GPL",
    url='https://github.com/ipselium/nsfds3',
    author="Cyril Desjouy",
    author_email="cyril.desjouy@univ-lemans.fr",
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "plotly",
                      "progressbar33", "rich", "h5py",],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    entry_points={
        'console_scripts': [
            'nsfds3 = nsfds3.main:main',
        ],
    }
)
