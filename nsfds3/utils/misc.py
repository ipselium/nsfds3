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
# Creation Date : 2022-05-18 - 22:03:45
# pylint: disable=too-few-public-methods
"""
-----------

Some tools used by the mesher.

-----------
"""

import sys as _sys
import datetime as _datetime
import itertools as _it
import numpy as _np
from libfds.cutils import yconsecutives, zconsecutives


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

    def __init__(self, stencil=3, ndim=3):

        if ndim not in [2, 3]:
            raise ValueError('dim must be 2 or 3')

        # 8 corners in 3d / 4 corners in 2d
        if ndim == 3:
            self.corners = [(stencil, stencil, stencil), (-stencil, stencil, stencil),
                            (-stencil, -stencil, stencil), (-stencil, -stencil, -stencil),
                            (stencil, -stencil, -stencil), (stencil, stencil, -stencil),
                            (stencil, -stencil, stencil), (-stencil, stencil, -stencil)]
        else:
            self.corners = [(stencil, stencil), (-stencil, stencil),
                            (-stencil, -stencil), (stencil, -stencil)]

        # 12 edges in 3d / 4 edges in 2d
        if ndim == 3:
            self.edges = [(stencil, stencil, 0), (0, stencil, stencil),
                          (stencil, 0, stencil), (-stencil, stencil, 0),
                          (stencil, -stencil, 0), (-stencil, -stencil, 0),
                          (0, -stencil, stencil), (0, stencil, -stencil),
                          (0, -stencil, -stencil), (-stencil, 0, stencil),
                          (stencil, 0, -stencil), (-stencil, 0, -stencil)]
        else:
            self.edges = [(stencil, 0), (-stencil, 0),
                          (0, stencil), (0, -stencil)]

        # 6 faces in 3d / 0 faces in 2d
        if ndim == 3:
            self.faces = [(stencil, 0, 0), (0, stencil, 0), (0, 0, stencil),
                          (-stencil, 0, 0), (0, -stencil, 0), (0, 0, -stencil)]
        else:
            self.faces = []

        # centered
        self.centered = [(0,) * ndim, ]

    def __call__(self, all=True):
        if all:
            return self.all
        return self.uncentered

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


def bc3d_tobc2d(bc, ax):
    """ Convert 3d bc to 2d. """
    return ''.join(b for i, b in enumerate(bc) if i not in [2 * ax, 2 * ax + 1])


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


def locations_to_cuboids_legacy(coordinates):
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


def groups(coordinates, only_conf=False):
    """ Return configuration and groups for cuboids searching... """

    if coordinates.ndim == 1:
        data = coordinates
    else:
        data = coordinates[:, -1]

    idx = _np.r_[0, _np.where(_np.diff(data) != 1)[0] + 1, len(data)]
    splits = _np.asarray([(data[i], data[j - 1]) for i, j in zip(idx, idx[1:])], dtype=_np.int16)
    confs = sorted(list(set((s[0], s[-1]) for s in splits)))
    if not only_conf:
        if coordinates.shape[1] == 3:
            consec = zconsecutives(splits, coordinates)
        elif coordinates.shape[1] == 2:
            consec = yconsecutives(splits, _np.ascontiguousarray(coordinates))
        coords = unique(consec)
        idx = [0, *_np.cumsum([len(list(value)) for key, value
                                     in _it.groupby(zip(*coords[:, :2].T))])]

        if coordinates.shape[1] == 3:
            groups = [coords[i:j, 2:] for i, j in zip(idx, idx[1:])]
        elif coordinates.shape[1] == 2:
            groups = [coords[i:j, 2] for i, j in zip(idx, idx[1:])]

        return confs, groups

    return confs


def locations_to_cuboids(coordinates):
    """ Search location of cuboid in a set of coordinates. """
    if coordinates.shape[1] == 3:
        return _locations_to_3d_cuboids(coordinates)

    if coordinates.shape[1] == 2:
        return _locations_to_2d_cuboids(coordinates)

    raise ValueError('Coordinates must have (x, 2) or (x, 3) shape')


def _locations_to_3d_cuboids(coordinates):
    cuboids = []
    zconfs, zgroups = groups(coordinates)

    for zconf, zgroup in zip(zconfs, zgroups):
        yconfs, ygroups = groups(zgroup)
        for yconf, ygroup in zip(yconfs, ygroups):
            xconfs = groups(ygroup, only_conf=True)
            cuboids.extend([{'origin': (i[0], j[0], k[0]),
                             'size': (i[1] - i[0] + 1, j[1] - j[0] + 1, k[1] - k[0] + 1)}
                            for i, j, k in [(x, yconf, zconf) for x in xconfs]])
    return cuboids


def _locations_to_2d_cuboids(coordinates):
    cuboids = []
    yconfs, ygroups = groups(coordinates)

    for yconf, ygroup in zip(yconfs, ygroups):
        xconfs = groups(ygroup, only_conf=True)
        cuboids.extend([{'origin': (i[0], j[0]),
                         'size': (i[1] - i[0] + 1, j[1] - j[0] + 1)}
                          for i, j in [(x, yconf) for x in xconfs]])
    return cuboids


def getsizeof(obj, seen=None, unit=None):
    """Recursively finds size of objects in bytes."""
    scale = 1e-3 if unit == 'k' else 1e-6 if unit == 'M' else 1e-9 if unit == 'G' else 1
    size = _sys.getsizeof(obj) * scale
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(getsizeof(v, seen=seen, unit=unit) for v in obj.values())
        size += sum(getsizeof(k, seen=seen, unit=unit) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += getsizeof(obj.__dict__, seen=seen, unit=unit)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(getsizeof(i, seen=seen, unit=unit) for i in obj)
    return size


def secs_to_dhms(secs):
    """ Convert seconds to years, months, days, hh:mm:ss."""

    dhms = _datetime.datetime(1, 1, 1) + _datetime.timedelta(seconds=secs)

    year, years = f'{dhms.year-1} year, ', f'{dhms.year-1} years, '
    month, months = f'{dhms.month-1} month, ', f'{dhms.month-1} months, '
    day, days = f'{dhms.day-1} day, ', f'{dhms.day-1} days, '
    h = f'{dhms.hour}:'
    m = f'{dhms.minute:02}:'
    s = f'{dhms.second:02}:'
    ms = f'{str(dhms.microsecond)[:2]}'

    return (year if dhms.year == 2 else years if dhms.year > 2 else '') + \
           (month if dhms.month == 2 else months if dhms.month > 2 else '') + \
           (day if dhms.day == 2 else days if dhms.day > 2 else '') + \
           (h if dhms.hour > 0 else '') + m + s + ms
