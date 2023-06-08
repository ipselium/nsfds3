#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2023 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
# Creation Date : 2022-06-09 - 23:00:01
"""
-----------

Some tools for cpgrid.

-----------
"""

def sign(x):
    """ Returns sign of x. """
    return -1 if x < 0 else 1


def buffer_bounds(bc, nbz):
    """ From bc, returns a list of indices corresponding to the limits of the buffer zone. """
    return [[sign(0.5 - j) * nbz if v == "A" else -j for j, v in enumerate(bc[i:i+2])]
             for i in range(0, len(bc), 2)]


def buffer_kwargs(bc, nbz, shape):
    """ Returns a dict containing origin, size and bc of the buffer zone.

    Parameters
    ----------
    bc : int
        Boundary conditions
    nbz : int
        size of the buffer zone
    shape : tuple
        size of the domain
    """
    bounds = buffer_bounds(bc, nbz)
    origin = [c[0] for c in bounds]
    size = [n + c[1] - o + 1 for n, c, o in zip(shape, bounds, origin)]
    return dict(origin=origin, size=size, env=shape, bc=bc)