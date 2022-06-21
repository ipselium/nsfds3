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
# Creation Date : 2022-05-18 - 22:03:45
# pylint: disable=too-few-public-methods
"""
-----------
DOCSTRING

-----------
"""

import sys as _sys
import itertools as _it
import more_itertools as _mit
import numpy as _np


class GeoMeta:
    """ Metaclasse for geometrical elements or sets. """

    count = 0

    @classmethod
    def count_reset(cls):
        """ Reset the count. """
        cls.count = 0

    @property
    def cls(self):
        """ Return class name. """
        return type(self).__name__

    def __str__(self):
        """ Custom __str__ following dataclass style. """
        if getattr(self, '__fargs__', None):
            return f"{type(self).__name__}({self.__fargs__()})"

        return f"{type(self).__name__}"

    def __repr__(self):
        return self.__str__()


class Schemes:
    """ Listing of schemes. """

    def __init__(self, stencil=3):

        # 8 corners
        self.corners = [(stencil, stencil, stencil), (-stencil, stencil, stencil),
                        (-stencil, -stencil, stencil), (-stencil, -stencil, -stencil),
                        (stencil, -stencil, -stencil), (stencil, stencil, -stencil),
                        (stencil, -stencil, stencil), (-stencil, stencil, -stencil)]
        # 12 edges
        self.edges = [(stencil, stencil, 0), (0, stencil, stencil),
                      (stencil, 0, stencil), (-stencil, stencil, 0),
                      (stencil, -stencil, 0), (-stencil, -stencil, 0),
                      (0, -stencil, stencil), (0, stencil, -stencil),
                      (0, -stencil, -stencil), (-stencil, 0, stencil),
                      (stencil, 0, -stencil), (-stencil, 0, -stencil)]
        # 6 faces
        self.faces = [(stencil, 0, 0), (0, stencil, 0), (0, 0, stencil),
                      (-stencil, 0, 0), (0, -stencil, 0), (0, 0, -stencil)]

        # centered
        self.centered = [(0, 0, 0), ]

    @property
    def all(self):
        """ Return all kind of schemes. """
        return self.centered + self.corners + self.edges + self.faces

    @property
    def uncentered(self):
        """ Return uncentered schemes. """
        return self.corners + self.edges + self.faces


def scheme_to_str(value):
    """ Convert scheme to str identifier. """
    return ''.join('c' if not v else 'm' if v < 0 else 'p' for v in value)


def bc3d_tobc2d(bc, axis):
    """ Convert 3d bc to 2d. """
    if axis == 0:
        idx = (0, 1)
    elif axis == 1:
        idx = (2, 3)
    elif axis == 2:
        idx = (3, 4)
    return ''.join(b for i, b in enumerate(bc) if i not in idx)


def consecutives(data, stepsize=1, coords=True):
    """ Search consecutives values in data. """

    idx = _np.r_[0, _np.where(_np.diff(data) != stepsize)[0] + 1, len(data)]
    splits = [data[i:j] for i, j in zip(idx, idx[1:])]
    #splits = _np.split(data, _np.where(_np.diff(data) != stepsize)[0] + 1)

    if coords:
        consec = _np.array(list(_it.chain(*[(s[0], s[-1]) * s.size for s in splits])),
                           dtype=_np.int16).reshape(data.size, 2)

    confs = sorted(list(set((s[0], s[-1]) for s in splits)))

    if coords:
        return confs, consec
    return confs


def unique(data, keys=None):
    """ Return unique elements of data """
    if not keys:
        keys = data.T[::-1]
    else:
        keys = [data[:, i] for i in keys]
    sorted_data = data[_np.lexsort(keys), :]
    row_mask = _np.append([True], _np.any(_np.diff(sorted_data, axis=0), 1))
    return sorted_data[row_mask]


def locations_to_cuboids(coordinates):
    """ Return cuboids filling coordinates. """

#    coordinates = (coordinates[:, 0].astype(_np.int16),
#                   coordinates[:, 1].astype(_np.int16),
#                   coordinates[:, 2].astype(_np.int16))

    zconfs, consec = consecutives(coordinates[:, 2])
    coords = unique(_np.concatenate((consec,
                                     coordinates[:, 0][:, _np.newaxis],
                                     coordinates[:, 1][:, _np.newaxis]), axis=1))
    idx = [0, *_np.cumsum([len(list(value)) for key, value
                                 in _it.groupby(zip(*coords[:, :2].T))])]
    zgroups = [coords[i:j, 2:] for i, j in zip(idx, idx[1:])]

    cuboids = []
    # Search cuboids in all groups
    for zconf, zgroup in zip(zconfs, zgroups):

        yconfs, consec = consecutives(zgroup[:, 1])
        coords = unique(_np.concatenate((consec,
                                         zgroup[:, 0][:, _np.newaxis]), axis=1))
        idx = [0, *_np.cumsum([len(list(value)) for key, value
                                    in _it.groupby(zip(*coords[:, :2].T))])]
        ygroups = [coords[i:j, 2] for i, j in zip(idx, idx[1:])]

        for yconf, ygroup in zip(yconfs, ygroups):
            xconfs = consecutives(ygroup, coords=False)
            cuboids.extend([{'origin': (i[0], j[0], k[0]),
                             'size': (i[1] - i[0] + 1, j[1] - j[0] + 1, k[1] - k[0] + 1)}
                            for i, j, k in [(x, yconf, zconf) for x in xconfs]])

    return cuboids


def getsizeof(obj, seen=None):
    """Recursively finds size of objects"""
    size = _sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(getsizeof(v, seen) for v in obj.values())
        size += sum(getsizeof(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += getsizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(getsizeof(i, seen) for i in obj)
    return size
