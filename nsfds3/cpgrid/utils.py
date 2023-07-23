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

import re
import numpy as _np


class GridError(Exception):
    """ Exception raised when grid parameters are wrong. """


def sign(x):
    """ Returns sign of x. """
    return -1 if x < 0 else 1


def parse_shape(shape):
    """ Check that grid shape is consistent, i.e. that each dimension does not exceed the max value of int16. """
    if len(shape) not in (2, 3):
        raise ValueError('shape: inconsistent dimension')
    if any(s > _np.iinfo(_np.int16).max for s in shape):
        raise GridError(f'shape: 1 dimension exceeds {_np.iinfo(_np.int16)}')
    return shape


def parse_bc(shape, bc):
    """ Parse boundary condition. """

    regex = [r'[^P]P..', r'P[^P]..', r'[^P]P....', r'P[^P]....',
                r'..[^P]P', r'..P[^P]', r'..[^P]P..', r'..P[^P]..',
                r'....[^P]P', r'....P[^P]',]

    if bc is None:
        bc = 'W' * len(shape) * 2

    if not isinstance(bc, str):
        raise ValueError('bc: str expected')
    else:
        bc = bc.upper()

    if len(bc) != 2 * len(shape):
        raise ValueError(f'bc: dim {len(2*shape)} expected')

    if not re.match(r'^[APW]*$', bc):
        raise ValueError(f"bc: must be combination of {2 * len(shape)} chars among 'APW'!")

    if any(re.match(r, bc) for r in regex):
        msg = "bc: 'P' must be on both sides of the domain,"
        msg += " i.e. '(PP....)'|'(..PP..)'|'(....PP)'"
        raise ValueError(msg)

    return bc


def parse_steps(shape, steps):
    """ Parse spacial steps. """

    if steps is None:
        steps = (1., ) * len(shape)

    if not isinstance(steps, tuple):
        raise ValueError('steps: tuple expected')

    if len(steps) != len(shape):
        raise ValueError(f'steps: length {len(shape)} expected')

    return steps


def parse_origin(shape, origin, bc, bz_n):
    """ Parse origin of the domain. """

    if origin is None:
        origin = tuple([bz_n if bc[2*i] == "A" else 0 for i in range(len(shape))])

    if not isinstance(origin, tuple):
        raise ValueError('origin: tuple expected')

    if len(origin) != len(shape):
        raise ValueError(f'origin: length {len(shape)} expected')

    if any(i0 >= N for i0, N in zip(origin, shape)):
        raise ValueError("origin: out of bounds")

    return origin


def parse_grid_parameters(shape, steps, origin, bc, bz_n):
    """ Check shape, origin, bc, and buffer zone. """
    shape = parse_shape(shape)
    steps = parse_steps(shape, steps)
    bc = parse_bc(shape, bc)
    origin = parse_origin(shape, origin, bc, bz_n)
    check_bz(shape, bz_n, bc)
    return shape, steps, origin, bc


def check_bz(shape, bz_n, bc):
    """ Check that buffer zone size is consistent. """
    if not all([n - bz_n * (bc[2*i:2*i + 2].count('A')) > 11 for i, n in enumerate(shape)]):
        raise GridError('bz: 1 of the dimension is too small.')


def buffer_bounds(bc, bz_n):
    """ From bc, returns a list of indices corresponding to the limits of the buffer zone. """
    return [[sign(0.5 - j) * bz_n if v == "A" else -j for j, v in enumerate(bc[i:i+2])]
             for i in range(0, len(bc), 2)]


def buffer_kwargs(bc, bz_n, shape):
    """ Returns a dict containing origin, size and bc of the buffer zone.

    Parameters
    ----------
    bc : int
        Boundary conditions
    bz_n : int
        size of the buffer zone
    shape : tuple
        size of the domain
    """
    bounds = buffer_bounds(bc, bz_n)
    origin = [c[0] for c in bounds]
    size = [n + c[1] - o + 1 for n, c, o in zip(shape, bounds, origin)]
    return dict(origin=origin, size=size, env=shape, bc=bc)