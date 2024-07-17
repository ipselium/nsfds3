#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2020 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of {name}
#
# {name} is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# {name} is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with {name}. If not, see <http://www.gnu.org/licenses/>.
#
# Creation Date : 2021-10-08 - 12:26:17
"""
-----------
DOCSTRING

-----------
"""

import numpy as np
from nsfds3.cpgrid import Obstacle

def cells2d(shape, ncells=40):
    """LBRT"""

    nx, ny = shape
    geo = []
    x0, y0 = 20, 20
    xref, yref = x0, y0
    xwidth, ywidth = 121, 121      # 13cm = 325
    xstreet, ystreet = 30, 30      # 1cm = 25pts

    for _ in range(ncells):
        xref = x0
        for _ in range(ncells):
            geo.append(Obstacle(origin=(xref, yref), size=(xwidth, ywidth), bc='WWWW', env=(nx, ny)))
            xref += xwidth + xstreet
        yref += ywidth + ystreet

    return geo


def gsource1(t):
    """Gaussian source."""
    t0 = 0.25e-3
    s = 0.8e-4   # 0.4e-3   (2ms en tout)
    return np.exp(- (t - t0)**2 / s**2)
