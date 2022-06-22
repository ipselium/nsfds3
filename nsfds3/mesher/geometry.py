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
# Creation Date : 2022-05-23 - 23:37:20
# pylint: disable=too-many-instance-attributes
"""
-----------

Module `domains` provides the `Obstacle` object used to define obstacles in the
computation domain.

It is the only object that is intended to be used directly by the user. The
other objects provided by the geometry module are used for the construction of
the calculation domains.

-----------
"""

__all__ = ['Obstacle', ]


import re as _re
import itertools as _it
import numpy as _np
from .utils import GeoMeta, scheme_to_str, bc3d_tobc2d


class BasicGeo(GeoMeta):
    """ Base class to describe 3d objects. """

    def __init__(self, origin, size, bc='WWWWWW', tag=None):

        self.origin = origin
        self.size = size
        self.bc = bc.upper()
        self.tag = tag

        if self.size[0]:
            self.ix = (self.origin[0], self.origin[0] + self.size[0] - 1)
        else:
            self.ix = (self.origin[0], self.origin[0])

        if self.size[1]:
            self.iy = (self.origin[1], self.origin[1] + self.size[1] - 1)
        else:
            self.iy = (self.origin[1], self.origin[1])

        if self.size[2]:
            self.iz = (self.origin[2], self.origin[2] + self.size[2] - 1)
        else:
            self.iz = (self.origin[2], self.origin[2])

        self.sx = slice(self.ix[0], self.ix[1] + 1)
        self.sy = slice(self.iy[0], self.iy[1] + 1)
        self.sz = slice(self.iz[0], self.iz[1] + 1)

        self.rx = range(self.ix[0], self.ix[1] + 1)
        self.ry = range(self.iy[0], self.iy[1] + 1)
        self.rz = range(self.iz[0], self.iz[1] + 1)

        self.slices = (self.sx, self.sy, self.sz)
        self.slices_inner = (slice(self.ix[0] + 1, self.ix[1]),
                             slice(self.iy[0] + 1, self.iy[1]),
                             slice(self.iz[0] + 1, self.iz[1]))
        self.ranges = (self.rx, self.ry, self.rz)

        self.bsize = _np.array([s for s in self.size if s]).cumprod()[-1] * 8e-6
        self.vertices = self._get_vertices()

    def _get_vertices(self):

        vertices = self.sort_vertices([tuple(i) for i in
                                            _it.product(self.ix, self.iy, self.iz)])

        return _np.array(list(vertices), dtype=_np.int16).T

    @staticmethod
    def sort_vertices(values):
        """ Sort vertices. """
        sort = [values[0], ]
        vertices = values[1:]

        while len(vertices):
            for v in vertices:
                if _np.count_nonzero((_np.array(v)
                                      - _np.array(sort[-1])) == 0) == len(values[0]) - 1:
                    sort.append(v)
                    vertices.remove(v)
                    break
        return sort

    @staticmethod
    def _fix_vertices(values):
        """ Fix flat cuboids. """
        axis = _np.where([v.min() == v.max() for v in values])[0]
        if axis.any():
            axis = axis[0]
            if axis == 0:
                idx = (4, 5, 6, 7)
            elif axis == 1:
                idx = (2, 3, 6, 7)
            elif axis == 2:
                idx = (1, 3, 5, 7)
            values = values.astype('float64')
            values[axis] = [v + 0.1 if i in idx else v
                                    for i, v in enumerate(values[axis])]
        return values

    def surfaces(self):
        """ Surfaces description for ax.plot_surface().

        Move to Obstacles ?
        """

        x = _np.array([[0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0]])
        y = _np.array([[0, 0, 1, 1, 0],
                       [0, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1]])
        z = _np.array([[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [0, 0, 1, 1, 0],
                       [0, 0, 1, 1, 0]])

        for i, v in enumerate([x, y, z]):
            v *= self.size[i]
            v += self.origin[i]

        return x, y, z

    @staticmethod
    def _parse_arg(other):
        if isinstance(other, tuple):
            return other
        return other.rx, other.ry, other.rz

    def intersects(self, other):
        """ Report whether self intersects other.
            Other can be a tuple or a BasicGeo.
        """
        cx, cy, cz = self.intersection(other)
        return any(cx) and any(cy) and any(cz)

    def intersection(self, other):
        """ Return intersection between self and other.
            Other can be a tuple or a BasicGeo.
        """
        rx, ry, rz = self._parse_arg(other)
        cx = set(self.rx).intersection(rx)
        cy = set(self.ry).intersection(ry)
        cz = set(self.rz).intersection(rz)
        return cx, cy, cz

    def difference(self, other):
        """ Return symmetric difference between self and other.
            Other can be a tuple or a BasicGeo.
        """
        rx, ry, rz = self._parse_arg(other)
        cx = set(self.rx).symmetric_difference(rx)
        cy = set(self.ry).symmetric_difference(ry)
        cz = set(self.rz).symmetric_difference(rz)
        return cx, cy, cz

    def issubset(self, other):
        """ Report whether other contains self.
            Other can be a tuple or a BasicGeo.
        """
        rx, ry, rz = self._parse_arg(other)
        cx = set(self.rx).issubset(rx)
        cy = set(self.ry).issubset(ry)
        cz = set(self.rz).issubset(rz)
        return cx and cy and cz

    def issuperset(self, other):
        """ Report whether self contains other.
            Other can be a tuple or a BasicGeo.
        """
        rx, ry, rz = self._parse_arg(other)
        cx = set(self.rx).issuperset(rx)
        cy = set(self.ry).issuperset(ry)
        cz = set(self.rz).issuperset(rz)
        return cx and cy and cz

    def box(self, shape, stencil=3):
        """ Return slices representing a box around the object. """
        sx = slice(max(0, self.ix[0] - stencil), min(shape[0], self.ix[1] + stencil + 1))
        sy = slice(max(0, self.iy[0] - stencil), min(shape[1], self.iy[1] + stencil + 1))
        sz = slice(max(0, self.iz[0] - stencil), min(shape[2], self.iz[1] + stencil + 1))
        return sx, sy, sz

    def to_2d(self, axis):
        """ Convert to 2d. """
        class2d = globals()[f"{self.cls}2d"]
        origin = tuple(o for i, o in enumerate(self.origin) if i != axis)
        size = tuple(s for i, s in enumerate(self.size) if i != axis)
        bc = bc3d_tobc2d(self.bc, axis)
        if self.tag:
            tag = tuple(t for i, t in enumerate(self.tag) if i != axis)
        else:
            tag = None
        return class2d(origin, size, bc=bc, tag=tag)

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, bc={self.bc}"


class BasicGeo2d(GeoMeta):
    """ Base class to describe 2d objects. """

    def __init__(self, origin, size, bc='WWWW', tag=None):

        self.sid = self.__class__.count
        self.__class__.count += 1

        self.origin = origin
        self.size = size
        self.bc = bc.upper()
        self.tag = tag

        self.ix = (self.origin[0], self.origin[0] + self.size[0] - 1)
        self.iy = (self.origin[1], self.origin[1] + self.size[1] - 1)

        self.sx = slice(self.ix[0], self.ix[1] + 1)
        self.sy = slice(self.iy[0], self.iy[1] + 1)

        self.rx = range(self.ix[0], self.ix[1] + 1)
        self.ry = range(self.iy[0], self.iy[1] + 1)

        self.slices = (self.sx, self.sy)
        self.slices_inner = (slice(self.ix[0] + 1, self.ix[1]),
                             slice(self.iy[0] + 1, self.iy[1]))
        self.ranges = (self.rx, self.ry)

        self.bsize = _np.array([s for s in self.size if s]).cumprod()[-1] * 8e-6
        self.vertices = self._get_vertices()

    def _get_vertices(self):
        vertices = BasicGeo.sort_vertices([tuple(i) for i
                                           in _it.product(self.ix, self.iy)])
        return _np.array(list(vertices), dtype=_np.int16).T

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, bc={self.bc}"


class Face(BasicGeo):
    """ Face of a cuboid. """

    # (axis, normal, bc index)
    sides = {'right': (0, 1, 0), 'left': (0, -1, 1),
             'back': (1, 1, 2), 'front': (1, -1, 3),
             'top': (2, 1, 4), 'bottom': (2, -1, 5)}

    opposites = {'right': 'left', 'left': 'right',
                 'back': 'front', 'front': 'back',
                 'top': 'bottom', 'bottom': 'top'}

    def __init__(self, origin, size, side, sid, bc='.', inner=False):

        super().__init__(origin, size)

        self.side = side
        self.sid = sid
        self.bc = bc
        self.opposite = self.opposites[side]
        self.axis, self.normal, self.index_n = self.sides[self.side]
        self.not_axis = tuple({0, 1, 2}.difference((self.axis, )))
        if inner:
            self.normal = - self.normal

        self.clamped = False        # is face attached to computation domain ?
        self.free = True            # free face
        self.overlapped = False     # face overlapped by another
        self.bounded = False
        self.loc = (self.ix, self.iy, self.iz)[self.axis][0]

    def update_boundary(self, domain):
        """ Update bound_cluster/bound_cdomain to take into account
            environment """

        # Check if this face is a subset of a boundary of the domain
        self.clamped = self.loc in [0, domain.shape[self.axis] - 1]

        # Check if this face is bounded by at least one other face in the
        # orthogonal direction
        self.bounded = [(face.axis, face.normal) for face in domain.faces
                        if face.axis != self.axis and face.sid != self.sid
                        and self.intersects(face)
                        and [len(i) > 1 for i in self.intersection(face)].count(True) == 1
                        and (self.loc in face.ranges[self.axis][1:-1])]

        # Check if another face overlaps this face
        self.overlapped = [face for face in domain.faces
                           if face.side == self.opposite and self.intersects(face)]

        # Check if another face covers entirely this face
        covered = [face for face in domain.faces
                   if face.side == self.opposite and self.issubset(face)]
        self.free = not self.clamped and not covered and not self.overlapped and not self.bounded

    def have_common_spatial_spread(self, other):
        """ Report whether self and other have common spatial spreading. """
        r_self = [r for i, r in enumerate(self.ranges) if i != self.axis]
        r_other = [r for i, r in enumerate(other.ranges) if i != self.axis]
        dim1 = set(r_self[0]).intersection(r_other[0])
        dim2 = set(r_self[1]).intersection(r_other[1])
        return dim1 and dim2

    def __fargs__(self):

        chars = ''
        if self.clamped:
            chars += "clamped"
        if self.overlapped:
            chars += "overlapped"
        if self.bounded:
            chars += "bounded"
        if self.free:
            chars += "free"

        s = f"origin={self.origin}, size={self.size}, "
        s += f"side={self.side}, sid={self.sid}, [{chars}]"

        return s


class Obstacle(BasicGeo):
    """ 3d Obstacle object. """

    def __init__(self, origin, size, bc='WWWWWW'):

        super().__init__(origin, size, bc)

        # Obstacles Identity
        self.sid = self.__class__.count
        self.__class__.count += 1
        self.faces = FaceSet(self)
        self.check()

    def check(self):
        """ Check if the obstacle is well defined. """
        if not self.is_valid_bc():
            self.sid -= 1
            raise ValueError(f"{self} : invalid bc")

        if not self.is_larger_enough():
            self.sid -= 1
            raise ValueError(f'{self} too small)')

    def is_larger_enough(self):
        """ Check if dimensions are larger than 5 points. """
        return all(s >= 5 for s in self.size)

    def is_valid_bc(self):
        """ Check if bc is a combination of 'ZWV'. """

        if _re.match(r'^[ZWV]+$', self.bc):
            return True

        return False


    def __iter__(self):
        return iter(self.faces)

    def __contains__(self, value):
        return value in self.faces._faces.values()

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, sid={self.sid}, bc={self.bc}"


class Obstacle2d(BasicGeo2d):
    """ 2d Obstacle object. """

    def __init__(self, origin, size, bc='WWWWWW', tag=None):

        super().__init__(origin, size, bc=bc, tag=tag)

        # Obstacles Identity
        self.sid = self.__class__.count
        self.__class__.count += 1

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, sid={self.sid}, bc={self.bc}"


class Domain(BasicGeo):
    """ 3d computation domain. """

    def __init__(self, origin, size, bc='WWWWWW', tag=(0, 0, 0)):

        super().__init__(origin, size, bc=bc, tag=tag)
        self.sid = 0
        self.faces = FaceSet(self, inner=True)
        self.scheme = scheme_to_str(self.tag)

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, scheme={self.scheme}"

    def __iter__(self):
        return iter(self.faces)


class Domain2d(BasicGeo2d):
    """ 2d computation domain. """

    def __init__(self, origin, size, bc='WWWWWW', tag=(0, 0)):

        super().__init__(origin, size, bc=bc, tag=tag)
        self.sid = 0
        self.scheme = scheme_to_str(self.tag)

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, scheme={self.scheme}"


class ObstacleSet(GeoMeta):
    """ Collection of Obstacle objects. """

    def __init__(self, shape, subs=None, stencil=3):
        self.subs = [] if not subs else subs
        self.shape = shape
        self.stencil = stencil
        self._check_obstacles()
        self.faces = FaceSet(self)
        for face in self.faces:
            face.update_boundary(self)

    def _check_obstacles(self):
        """ Check that all obstacles are valid ones. In particular, check if :

                * all objects are Obstacles,
                * all obstacles are inside the domain,
                * all obstacles have valid location considering bounds of the domain,
                * two obstacles intersect,
                * two obstacles have faces too close,
                * (TODO) faces overlaps by more than stencil points.

            Raises an exception if at least one Obstacle is not properly
            defined. """

        if self.subs:
            if not all(isinstance(sub, Obstacle) for sub in self):
                raise ValueError('subs must only contain Obstacle objects')

        for obs in self:
            if self.is_close_to_bound(obs):
                msg = f'{obs} too close from boundary ({self.stencil} pts minimum)'
                raise ValueError(msg)
            if self.is_out_of_bounds(obs):
                raise ValueError(f'{obs} is out of bounds')

        for obs1, obs2 in _it.combinations(self, r=2):
            if obs1.intersects(obs2):
                if not ([len(x) == 1 for x in obs1.intersection(obs2)]).count(True) == 1:
                    msg = f'Superimposed (Wrong):\n\t- {obs1}\n\t- {obs2}'
                    raise ValueError(msg)
            else:
                for face1 in obs1:
                    face2 = getattr(obs2.faces, face1.opposite)
                    if face1.have_common_spatial_spread(face2) \
                            and abs(face2.loc - face1.loc) < 2 * self.stencil:
                        msg = f'Too close:\n\t- {face1}\n\t- {face2}'
                        raise ValueError(msg)

    def is_out_of_bounds(self, obs):
        """Check if subdomain is out of bounds. """
        return any(s.start not in range(n) or s.stop - 1 not in range(n)
                   for s, n in zip(obs.slices, self.shape))

    def is_close_to_bound(self, obs):
        """ Check if the obstacle has location compatible with the stencil. """
        return any(s.start in range(1, 2 * self.stencil + 1) or
                         s.stop - 1 in range(n - 1 - 2 * self.stencil, n - 1)
                         for s, n in zip(obs.slices, self.shape))

    def __getitem__(self, n):
        return self.subs[n]

    def __iter__(self):
        return iter(self.subs)

    def __len__(self):
        return len(self.subs)

    def __fargs__(self):
        s = f'{len(self)} elements:'
        for edge in self.subs:
            s += f'\n\t- {edge}'
        return s


class FaceSet(GeoMeta):
    """ Collection of Face objects. """

    def __init__(self, subs, inner=False):
        self.subs = [] if not subs else subs

        if isinstance(self.subs, (Obstacle, Domain)):
            self.left = Face(origin=(subs.ix[0], subs.iy[0], subs.iz[0]),
                             size=(0, subs.size[1], subs.size[2]),
                             side='left', sid=subs.sid,
                             bc=subs.bc[0], inner=inner)
            self.right = Face(origin=(subs.ix[1], subs.iy[0], subs.iz[0]),
                              size=(0, subs.size[1], subs.size[2]),
                              side='right', sid=subs.sid,
                              bc=subs.bc[1], inner=inner)
            self.front = Face(origin=(subs.ix[0], subs.iy[0], subs.iz[0]),
                              size=(subs.size[0], 0, subs.size[2]),
                              side='front', sid=subs.sid,
                              bc=subs.bc[2], inner=inner)
            self.back = Face(origin=(subs.ix[0], subs.iy[1], subs.iz[0]),
                             size=(subs.size[0], 0, subs.size[2]),
                             side='back', sid=subs.sid,
                             bc=subs.bc[3], inner=inner)
            self.bottom = Face(origin=(subs.ix[0], subs.iy[0], subs.iz[0]),
                               size=(subs.size[0], subs.size[1], 0),
                               side='bottom', sid=subs.sid,
                               bc=subs.bc[4], inner=inner)
            self.top = Face(origin=(subs.ix[0], subs.iy[0], subs.iz[1]),
                            size=(subs.size[0], subs.size[1], 0),
                            side='top', sid=subs.sid,
                            bc=subs.bc[5], inner=inner)

            self._faces = {'left': self.left, 'right': self.right,
                           'front': self.front, 'back': self.back,
                           'bottom': self.bottom, 'top': self.top}

        elif isinstance(self.subs, ObstacleSet):
            faces = [face for subs in self.subs for face in subs]
            self._faces = {key: [face for face in faces if face.side == key]
                           for key in Face.sides}
            self.__dict__.update(self._faces)

        elif self.subs:
            raise ValueError('subs must be Obstacle or ObstacleSet object')

        else:
            self._faces = {key: [] for key in Face.sides}

    @property
    def overlapped(self):
        """ Return all pairs of faces that overlap. """
        return [(face1, face2) for face1, face2 in _it.combinations(self, r=2)
                if face1.side == face2.opposite and face1.intersects(face2) and
                face1.overlapped and face2.overlapped]

    @property
    def common_edge(self):
        """ Return all pairs of faces that have a common edge. """
        return [(face1, face2) for face1, face2 in _it.combinations(self, r=2)
                 if face1.side == face2.side and face1.loc == face2.loc
                 and face1.intersects(face2)
                 and not face1.clamped and not face2.clamped]

    def items(self):
        """ A set-like object providing a view on self items. """
        return self._faces.items()

    def __iter__(self):
        if isinstance(self.subs, (Obstacle, Domain)):
            return iter(self._faces.values())
        return _it.chain(*self._faces.values())

    def __len__(self):
        if isinstance(self.subs, (Obstacle, Domain)):
            return len(self._faces)
        return len(tuple(_it.chain(*self._faces.values())))

    def __getitem__(self, key):
        return self._faces[key]

    def __fargs__(self):
        s = f'{len(self)} elements:'
        for sub in self.subs:
            s += f'\n\t- {sub}'
            if isinstance(self.subs, (ObstacleSet)):
                for face in sub:
                    s += f'\n\t\t- {face}'
        return s
