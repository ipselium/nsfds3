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
from nsfds3.utils.misc import GeoMeta, scheme_to_str


class BasicGeo(GeoMeta):
    """ Base class to describe 3d objects. """

    def __init__(self, origin, size, bc=None, tag=None):

        if len(origin) != len(size):
            raise ValueError('origin and size must have the same size')

        self.origin = origin
        self.size = size
        self.tag = tag
        self.volumic = len(origin) == 3
        if not bc:
            self.bc = 'WWWWWW' if self.volumic else 'WWWW'
        else:
            self.bc = bc.upper()

        if self.size[0]:
            self.ix = (self.origin[0], self.origin[0] + self.size[0] - 1)
        else:
            self.ix = (self.origin[0], self.origin[0])

        if self.size[1]:
            self.iy = (self.origin[1], self.origin[1] + self.size[1] - 1)
        else:
            self.iy = (self.origin[1], self.origin[1])

        self.sx = slice(self.ix[0], self.ix[1] + 1)
        self.sy = slice(self.iy[0], self.iy[1] + 1)

        self.rx = range(self.ix[0], self.ix[1] + 1)
        self.ry = range(self.iy[0], self.iy[1] + 1)

        if self.volumic:
            if self.size[2]:
                self.iz = (self.origin[2], self.origin[2] + self.size[2] - 1)
            else:
                self.iz = (self.origin[2], self.origin[2])
            self.sz = slice(self.iz[0], self.iz[1] + 1)
            self.rz = range(self.iz[0], self.iz[1] + 1)

            self.slices = (self.sx, self.sy, self.sz)
            self.slices_inner = (slice(self.ix[0] + 1, self.ix[1]),
                                 slice(self.iy[0] + 1, self.iy[1]),
                                 slice(self.iz[0] + 1, self.iz[1]))
            self.ranges = (self.rx, self.ry, self.rz)
            self.coords = (self.ix, self.iy, self.iz)
        else:
            self.slices = (self.sx, self.sy)
            self.slices_inner = (slice(self.ix[0] + 1, self.ix[1]),
                                 slice(self.iy[0] + 1, self.iy[1]))
            self.ranges = (self.rx, self.ry)
            self.coords = (self.ix, self.iy)

        self.bsize = _np.array([s for s in self.size if s]).cumprod()[-1] * 8e-6
        self.vertices = self._get_vertices()

    def _get_vertices(self):

        vertices = self.sort_vertices([tuple(i) for i in _it.product(*self.coords)])
        return _np.array(vertices, dtype=_np.int16).T

    @staticmethod
    def sort_vertices(values):
        """ Sort vertices. """
        sort = [values[0], ]
        vertices = values[1:]

        n = 0
        while len(vertices):
            for v in vertices:
                if _np.count_nonzero((_np.array(v)
                                      - _np.array(sort[-1])) == 0) == len(values[0]) - 1:
                    sort.append(v)
                    vertices.remove(v)
                    break
            n += 1
            if n > len(values)**2:
                sort = values[:]
                break
        # append first value to the end to be able to plot all contour...
        sort.append(sort[0])
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

    def _parse_arg(self, other):
        if isinstance(other, tuple):
            return other
        if self.volumic:
            return other.rx, other.ry, other.rz
        return other.rx, other.ry

    def _check_dim(self, other):
        if self.volumic != other.volumic:
            raise ValueError('Objects must have same dimensions')

    def intersects(self, other):
        """ Report whether self intersects other.
            Other can be a tuple or a BasicGeo.
        """
        self._check_dim(other)
        if self.volumic:
            cx, cy, cz = self.intersection(other)
            return any(cx) and any(cy) and any(cz)

        cx, cy = self.intersection(other)
        return any(cx) and any(cy)

    def intersection(self, other):
        """ Return intersection between self and other.
            Other can be a tuple or a BasicGeo.
        """
        self._check_dim(other)
        if self.volumic:
            rx, ry, rz = self._parse_arg(other)
            cx = set(self.rx).intersection(rx)
            cy = set(self.ry).intersection(ry)
            cz = set(self.rz).intersection(rz)
            return cx, cy, cz

        rx, ry = self._parse_arg(other)
        cx = set(self.rx).intersection(rx)
        cy = set(self.ry).intersection(ry)
        return cx, cy

    def difference(self, other):
        """ Return symmetric difference between self and other.
            Other can be a tuple or a BasicGeo.
        """
        self._check_dim(other)
        if self.volumic:
            rx, ry, rz = self._parse_arg(other)
            cx = set(self.rx).symmetric_difference(rx)
            cy = set(self.ry).symmetric_difference(ry)
            cz = set(self.rz).symmetric_difference(rz)
            return cx, cy, cz

        rx, ry = self._parse_arg(other)
        cx = set(self.rx).symmetric_difference(rx)
        cy = set(self.ry).symmetric_difference(ry)
        return cx, cy

    def issubset(self, other):
        """ Report whether other contains self.
            Other can be a tuple or a BasicGeo.
        """
        self._check_dim(other)
        if self.volumic:
            rx, ry, rz = self._parse_arg(other)
            cx = set(self.rx).issubset(rx)
            cy = set(self.ry).issubset(ry)
            cz = set(self.rz).issubset(rz)
            return cx and cy and cz

        rx, ry = self._parse_arg(other)
        cx = set(self.rx).issubset(rx)
        cy = set(self.ry).issubset(ry)
        return cx and cy

    def issuperset(self, other):
        """ Report whether self contains other.
            Other can be a tuple or a BasicGeo.
        """
        self._check_dim(other)
        if self.volumic:
            rx, ry, rz = self._parse_arg(other)
            cx = set(self.rx).issuperset(rx)
            cy = set(self.ry).issuperset(ry)
            cz = set(self.rz).issuperset(rz)
            return cx and cy and cz

        rx, ry = self._parse_arg(other)
        cx = set(self.rx).issuperset(rx)
        cy = set(self.ry).issuperset(ry)
        return cx and cy

    def box(self, shape, stencil=11):
        """ Return slices representing a box around the object. """
        midstencil = int((stencil - 1) / 2)
        sx = slice(max(0, self.ix[0] - midstencil), min(shape[0], self.ix[1] + midstencil + 1))
        sy = slice(max(0, self.iy[0] - midstencil), min(shape[1], self.iy[1] + midstencil + 1))
        if self.volumic:
            sz = slice(max(0, self.iz[0] - midstencil), min(shape[2], self.iz[1] + midstencil + 1))
            return sx, sy, sz
        return sx, sy

    def flatten(self, axis):
        """ Return flat version of the object. """
        cls = globals()[f"{self.cls}"]

        if not self.volumic:
            raise ValueError(f'{cls.__name__} already flat')

        origin = tuple(o for i, o in enumerate(self.origin) if i != axis)
        size = tuple(s for i, s in enumerate(self.size) if i != axis)
        tag = tuple(s for i, s in enumerate(self.tag) if i != axis) if self.tag else None
        bc = ''.join([v for i, v in enumerate(self.bc) if i not in [2*axis, 2*axis + 1]])

        return cls(origin, size, bc=bc, tag=tag)

    def __contains__(self, coords):
        return all(c[0] <= p <= c[1] for p, c in zip(coords, self.coords))

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, bc={self.bc}"


class Obstacle(BasicGeo):
    """ 3d Obstacle object. """

    def __init__(self, origin, size, bc=None, tag=None):

        super().__init__(origin, size, bc=bc, tag=tag)

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
        if not isinstance(value, (tuple, list)):
            raise ValueError('Value must be a tuple')
        return value in self.faces._faces.values()

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, sid={self.sid}, bc={self.bc}"


class BasicSet(GeoMeta):
    """ Base class to describe sets of objects. """

    def __init__(self, shape, subs=None, stencil=3):
        self.subs = [] if not subs else subs
        self.shape = shape
        self.stencil = stencil

    @property
    def inner_objects(self):
        """ Return an new instance without objects located at the limits of the domain. """
        if len(self.shape) == 3:
            subs = [sub for sub in self
                    if sub.ix[0] != 0 and sub.iy[0] != 0 and sub.iz[0] != 0
                    and sub.ix[1] != self.shape[0] - 1
                    and sub.iy[1] != self.shape[1] - 1
                    and sub.iz[1] != self.shape[2] - 1]
        else:
            subs = [sub for sub in self
                    if sub.ix[0] != 0 and sub.iy[0] != 0
                    and sub.ix[1] != self.shape[0] - 1
                    and sub.iy[1] != self.shape[1] - 1]
        return DomainSet(self.shape, subs=subs, stencil=self.stencil)

    def __getitem__(self, n):
        return self.subs[n]

    def __iter__(self):
        return iter(self.subs)

    def __len__(self):
        return len(self.subs)

    def __fargs__(self):
        s = f'shape={self.shape} -- {len(self)} elements:'
        for edge in self.subs:
            s += f'\n\t- {edge}'
        return s


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
        self.free = True            # is face free ?
        self.overlapped = False     # is face overlapped by another ?
        self.bounded = False
        self.loc = self.coords[self.axis][0]

    def update_boundary(self, domain):
        """ Update bound_cluster/bound_cdomain to take into account
            environment """

        # Check if this face is a subset of a boundary of the domain
        self.clamped = self.loc in [0, domain.shape[self.axis] - 1]

        # Check if another face overlaps this face
        self.overlapped = [face for face in domain.faces
                           if face.side == self.opposite and self.intersects(face)]

        # Check if this face is bounded by at least one other face in the
        # orthogonal direction
        self.bounded = [(face.axis, face.normal) for face in domain.faces
                        if face.axis != self.axis and face.sid != self.sid
                        and self.intersects(face)
                        #                        and [len(i) > 1 for i in self.intersection(face)].count(True) == 1
                        #                        and (self.loc in face.ranges[self.axis][1:-1])]
                        and (self.loc in face.ranges[self.axis])
                        and not self.overlapped]


        # Check if another face covers entirely this face
        covered = [face for face in domain.faces
                   if face.side == self.opposite and self.issubset(face)]
        self.free = not self.clamped and not covered and not self.overlapped and not self.bounded

    def have_common_spatial_spread(self, other):
        """ Report whether self and other have common spatial spreading. """
        r_self = [r for i, r in enumerate(self.ranges) if i != self.axis]
        r_other = [r for i, r in enumerate(other.ranges) if i != self.axis]
        dim1 = set(r_self[0]).intersection(r_other[0])
        if len(self.origin) == 3:
            dim2 = set(r_self[1]).intersection(r_other[1])
            return dim1 and dim2
        return dim1

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


class FaceSet(GeoMeta):
    """ Collection of Face objects. """

    def __init__(self, subs, inner=False):
        self.subs = [] if not subs else subs

        if isinstance(self.subs, (Obstacle, Domain)):
            if self.subs.volumic:
                self._faces_volumic(subs, inner)
            else:
                self._faces_flat(subs, inner)

        elif isinstance(self.subs, (ObstacleSet, DomainSet)):
            faces = [face for subs in self.subs for face in subs]
            self._faces = {key: [face for face in faces if face.side == key]
                           for key in Face.sides}
            self.__dict__.update(self._faces)

        elif self.subs:
            raise ValueError('subs must be Obstacle or ObstacleSet object')

        else:
            self._faces = {key: [] for key in Face.sides}

    def _faces_volumic(self, subs, inner):
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

    def _faces_flat(self, subs, inner):
        self.left = Face(origin=(subs.ix[0], subs.iy[0]),
                         size=(0, subs.size[1]),
                         side='left', sid=subs.sid,
                         bc=subs.bc[0], inner=inner)
        self.right = Face(origin=(subs.ix[1], subs.iy[0]),
                          size=(0, subs.size[1]),
                          side='right', sid=subs.sid,
                          bc=subs.bc[1], inner=inner)
        self.front = Face(origin=(subs.ix[0], subs.iy[0]),
                          size=(subs.size[0], 0),
                          side='front', sid=subs.sid,
                          bc=subs.bc[2], inner=inner)
        self.back = Face(origin=(subs.ix[0], subs.iy[1]),
                         size=(subs.size[0], 0),
                         side='back', sid=subs.sid,
                         bc=subs.bc[3], inner=inner)

        self._faces = {'left': self.left, 'right': self.right,
                        'front': self.front, 'back': self.back}

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

    @property
    def not_clamped(self):
        """ Return faces that are not clamped to global domain. """
        return [face for face in self if not face.clamped]

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


class Domain(BasicGeo):
    """ 3d computation domain. """

    def __init__(self, origin, size, bc=None, tag=None):

        super().__init__(origin, size, bc=bc, tag=tag)
        self.tag = tag if tag else (0, ) * len(origin)
        self.sid = 0
        self.faces = FaceSet(self, inner=True)
        self.scm_du = scheme_to_str(self.tag)
        self.scm_fx = self.scm_du[0]
        self.scm_fy = self.scm_du[1]
        if self.volumic:
            self.scm_fz = self.scm_du[2]

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, scheme={self.scm_du}"

    def __iter__(self):
        return iter(self.faces)


class DomainSet(BasicSet):
    """ Collection of Domain objects. """

    def __init__(self, shape, subs=None, stencil=3):
        super().__init__(shape, subs, stencil)
        self.faces = FaceSet(self)


class ObstacleSet(BasicSet):
    """ Collection of Obstacle objects. """

    def __init__(self, shape, subs=None, stencil=3):
        super().__init__(shape, subs, stencil)
        self._check_config()
        self.faces = FaceSet(self)
        for face in self.faces:
            face.update_boundary(self)

    def _check_config(self):
        """ Check that all obstacles are valid ones. In particular, check if :

                * all objects are Obstacles,
                * all objects are flat or not
                * all obstacles are inside the domain,
                * all obstacles have valid location considering bounds of the domain,
                * two obstacles intersect,
                * two obstacles have faces too close,
                * (TODO) faces overlaps by more than stencil points.

            Raises an exception if at least one Obstacle is not properly
            defined. """

        if not all(isinstance(sub, Obstacle) for sub in self):
            raise ValueError('subs must only contain Obstacle objects')

        if len(set(sub.volumic for sub in self)) > 1:
            raise ValueError('All elements must be either 2D or 3D')

        if len(self.shape) == 3:
            self.volumic = True
        elif len(self.shape) == 2:
            self.volumic = False
        else:
            raise ValueError('Wrong size for shape')

        if self.subs:
            if (self.volumic and not self[0].volumic) \
               or (not self.volumic and self[0].volumic):
                raise ValueError("set and obstacle shapes don't mach")

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

    def flatten(self, axis, index=0):
        """ Return a flat version of the object. """
        if not self.volumic:
            raise TypeError('Already flat')

        obstacles = []
        for obs in self:
            if index in obs.ranges[axis]:
                obstacles.append(obs.flatten(axis))

        shape = tuple(o for i, o in enumerate(self.shape) if i != axis)
        obsset = ObstacleSet(shape=shape, subs=obstacles, stencil=self.stencil)
        obsset.__volumic = self

        return obsset

    def unflatten(self):
        """ Return a volumic version of flattened object. """
        instance = getattr(self, f'_{type(self).__name__}__volumic', None)
        if not instance:
            raise TypeError('The object does not have a volumic version')
        return instance
