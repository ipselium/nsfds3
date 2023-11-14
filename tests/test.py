#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2020 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
# Creation Date : 2022-07-21 - 09:00:24
"""
The module `data` provides some helper class or function to retrieve data from nsfds3 simulations.
"""


from nsfds3.graphics import MPLViewer
from nsfds3.utils import get_objects
from pathlib import Path

path = Path('/Users/cdesjouy/Documents/dev/apps/nsfds3/tests/configurations/None')
filename = Path('pulse_2d')

cfg2d, msh2d = get_objects(path, filename)
viewer2d = MPLViewer(cfg2d, msh2d, path / filename.with_suffix('.hdf5'))

msh2d.show(N=4, buffer=True, domains=False,
            kwargs_obstacles=dict(hatch='/', facecolor='r'),
            kwargs_buffer=dict(hatch='/', fill=True, facecolor='b', alpha=0.1)
            )