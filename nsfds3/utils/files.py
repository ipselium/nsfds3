#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2019 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of nsfds3
#
# nsfds3 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# nsfds3 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with nsfds3. If not, see <http://www.gnu.org/licenses/>.
#
# Creation Date : 2019-03-21 - 23:43:11
"""
-----------

Utils : Files

-----------
"""

import os
import sys
import pickle
import pathlib
from nsfds3.cpgrid import templates as _tplt


def get_func(module, name):
    """ Get obstacle from custom file or fdgrid templates. """

    if os.path.isfile(module):
        sys.path.append(os.path.dirname(module))
        custom = __import__(os.path.basename(module).split('.')[0])
    else:
        custom = _tplt.TestCases

    try:
        return getattr(custom, name)
    except AttributeError:
        return None


def get_wall_function(cfg, name):
    """ Get function to apply to wall source."""

    try:
        sys.path.append(os.path.dirname(cfg.geofile))
        custom = __import__(os.path.basename(cfg.geofile).split('.')[0])
        func = getattr(custom, name)
    except (AttributeError, ImportError):
        func = None
    return func

def get_objects(path, fname):

    path = pathlib.Path(path)
    fname = pathlib.Path(fname)

    with open(path / fname.with_suffix('.cfg'), 'rb') as f:
        cfg = pickle.load(f, encoding='bytes')

    with open(path / fname.with_suffix('.msh'), 'rb') as f:
        msh = pickle.load(f, encoding='bytes')

    return cfg, msh
