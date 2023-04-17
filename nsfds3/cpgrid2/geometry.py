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
# Creation Date : 2023-04-06 - 09:45:13
"""
-----------
DOCSTRING

-----------
"""

import itertools as _it
import numpy as _np

class Box:

    _axnames = 'x', 'y', 'z'

    def __init__(self, origin, size, bc=None, env=None, inner=False):

        self.origin = origin
        self.size = size
        self.bc = bc
        self._env = env
        self.inner = inner
        self.ndim = len(origin)

        self._set_bc()
        self._check_args()
        self._set_rn()
        self._set_sn()
        self._set_cn()
        self._set_rin()

        self.indices = tuple(_it.product(*[list(r) for r in self.rn]))

        self.vertices = self._get_vertices()

    def _get_vertices(self):

        vertices = self.sort_vertices([tuple(i) for i in _it.product(*self.cn)])
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

    @property
    def env(self):
        """ Return enveloppe of the object."""
        if not self._env:
            self._env = tuple(o + s + 5 for o, s in zip(self.origin, self.size))
        return self._env

    @env.setter
    def env(self, value):
        if len(value) != len(self.origin):
            raise ValueError('env must be of the same dimension as origin')
        self._env = tuple(value)

    def inner_indices(self, ax):
        """ Return indices. inner points except along axis (int)"""
        rn = list(self.rin)
        rn[ax] = self.rn[ax]
        return tuple(_it.product(*[list(r) for r in rn]))

    def intersection(self, other):
        """ Return intersection (point coordinate) between self and other."""
        return tuple(set(self.indices).intersection(other.indices))

    def intersects(self, other):
        """ Report whether self intersects other."""
        out = []
        for s, o in zip(self.rn, other.rn):
            out.append(set(s).intersection(o))
        return all(tuple(out))

    def difference(self, other):
        """ Return symmetric difference (point coordinates) between self and other."""
        return tuple(set(self.indices).symmetric_difference(other.indices))

    def issubset(self, other):
        """ Report whether other contains self."""
        out = []
        for s, o in zip(self.rn, other.rn):
            out.append(set(s).issubset(o))
        return all(tuple(out))

    def issuperset(self, other):
        """ Report whether self contains other."""
        out = []
        for s, o in zip(self.rn, other.rn):
            out.append(set(s).issuperset(o))
        return all(tuple(out))

    def box(self, N=5):
        """ Returns a Box extending N points around the object."""

        out = []
        for r, s in zip(self.rn, self.env):
            # Pour periodic, start peut aller dans le négatif et stop, revenir vers 0, 1, 2... !
            out.append([max(0, r.start - N), min(s, r.stop + N)])

        origin = tuple(c[0] for c in out)
        size = tuple(s[1] - s[0] for s in out)

        return Box(origin=origin, size=size, bc='X' * 2 * len(self.origin), env=self.env)

    def flatten(self, axis):
        """ Return flat version of the object."""
        cls = type(self)
        if self.ndim == 2:
            raise ValueError(f'{cls.__name__} already flat')

        origin = tuple(o for i, o in enumerate(self.origin) if i != axis)
        size = tuple(s for i, s in enumerate(self.size) if i != axis)
        bc = ''.join([v for i, v in enumerate(self.bc) if i not in [2*axis, 2*axis + 1]])

        return cls(origin, size, bc=bc)

    def _check_args(self):
        """ Check input arguments."""

        if any(len(self.bc) != 2 * len(s) for s in [self.origin, self.size, self.env]):
            raise ValueError('origin, size, env and bc must have coherent dimensions')

        if any(s == 0 for s in self.size):
            raise ValueError('Size of the object must be at least 1')

    def _set_bc(self):
        """ Set bc. Will be len(origin) * 'W' by default."""
        if not self.bc:
            self.bc = 'WWWWWW' if self.ndim == 3 else 'WWWW'
        else:
            self.bc = self.bc.upper()

    def _set_rn(self):
        """ Set ranges (rx, ry [, rz])."""
        self.rn = ()
        for n, i in zip(self._axnames, range(self.ndim)):
            setattr(self, f'r{n}', range(self.origin[i], self.origin[i] + self.size[i]))
            self.rn += (getattr(self, f'r{n}'), )

    def _set_rin(self):
        """ Set inner ranges (rix, riy [, riz])."""
        self.rin = ()
        for n, i in zip(self._axnames, range(self.ndim)):
            if self.size[i] <= 1:
                setattr(self, f'ri{n}', range(self.origin[i], self.origin[i] + self.size[i]))
            else:
                setattr(self, f'ri{n}', range(self.origin[i] + 1, self.origin[i] + self.size[i] - 1))
            self.rin += (getattr(self, f'ri{n}'), )

    def _set_sn(self):
        """ Set slices (sx, sy [, sz])."""
        self.sn = ()
        for n, i in zip(self._axnames, range(self.ndim)):
            setattr(self, f's{n}', slice(self.origin[i], self.origin[i] + self.size[i]))
            self.sn += (getattr(self, f's{n}'), )

    def _set_cn(self):
        """ Set coordinates (cx, cy [, cz])."""
        self.cn = ()
        for n, r in zip(self._axnames, self.rn):
            setattr(self, f'c{n}', (r[0], r[-1]))
            self.cn += (getattr(self, f'c{n}'), )

    def __contains__(self, other):
        return other.issubset(self)

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, bc={self.bc}"

    def __str__(self):
        """ Custom __str__ following dataclass style. """
        if getattr(self, '__fargs__', None):
            return f"{type(self).__name__}({self.__fargs__()})"

        return f"{type(self).__name__}"

    def __repr__(self):
        return self.__str__()


class Cuboid(Box):

    count = 0

    @classmethod
    def count_reset(cls):
        """ Reset the count. """
        cls.count = 0

    def __init__(self, origin, size, bc=None, env=None, inner=False):
        super().__init__(origin, size, bc=bc, env=env, inner=inner)

        # Object id
        self.sid = self.__class__.count
        self.__class__.count += 1

        # Faces
        if self.ndim == 3:
            self._set_faces_volumic()
        else:
            self._set_faces_flat()

    @property
    def description(self):
        # Note : clamped and covered are the same ... Maybe different when P bc...
        attributes = dict(clamped='C', bounded='b',
                          free='f', colinear='I',
                          overlapped='o', covered='0')

        chars = [''.join([attributes[c] for c in attributes.keys() if getattr(f, c)])
                 for f in self.faces]
        return '/'.join(chars)

    @property
    def env(self):
        return super().env

    @env.setter
    def env(self, value):

        if len(value) != len(self.origin):
            raise ValueError('env must be of the same dimension as origin')

        self._env = tuple(value)

        for face in self.faces:
            face.env = tuple(value)
            face.update_indices_u()

    def _set_faces_volumic(self):
        """ Set faces for cuboid. """
        self.face_left = Face(origin=(self.cx[0], self.cy[0], self.cz[0]),
                              size=(1, self.size[1], self.size[2]),
                              side='left', bc=self.bc[0],
                              sid=self.sid, inner=self.inner)

        self.face_right = Face(origin=(self.cx[1], self.cy[0], self.cz[0]),
                               size=(1, self.size[1], self.size[2]),
                               side='right', bc=self.bc[1],
                               sid=self.sid, inner=self.inner)


        self.face_front = Face(origin=(self.cx[0], self.cy[0], self.cz[0]),
                               size=(self.size[0], 1, self.size[2]),
                               side='front', bc=self.bc[2],
                               sid=self.sid, inner=self.inner)

        self.face_back = Face(origin=(self.cx[0], self.cy[1], self.cz[0]),
                              size=(self.size[0], 1, self.size[2]),
                              side='back', bc=self.bc[3],
                              sid=self.sid, inner=self.inner)

        self.face_bottom = Face(origin=(self.cx[0], self.cy[0], self.cz[0]),
                                size=(self.size[0], self.size[1], 1),
                                side='bottom', bc=self.bc[4],
                                sid=self.sid, inner=self.inner)

        self.face_top = Face(origin=(self.cx[0], self.cy[0], self.cz[1]),
                             size=(self.size[0], self.size[1], 1),
                             side='top', bc=self.bc[5],
                             sid=self.sid, inner=self.inner)

        self.faces = (self.face_left, self.face_right,
                      self.face_front, self.face_back,
                      self.face_bottom, self.face_top)

    def _set_faces_flat(self):
        """ Set edges forrectangle. """
        self.face_left = Face(origin=(self.cx[0], self.cy[0]),
                              size=(1, self.size[1]),
                              side='left', bc=self.bc[0],
                              sid=self.sid, inner=self.inner)

        self.face_right = Face(origin=(self.cx[1], self.cy[0]),
                               size=(1, self.size[1]),
                               side='right',bc=self.bc[1],
                               sid=self.sid, inner=self.inner)

        self.face_front = Face(origin=(self.cx[0], self.cy[0]),
                               size=(self.size[0], 1),
                               side='front', bc=self.bc[2],
                               sid=self.sid, inner=self.inner)

        self.face_back = Face(origin=(self.cx[0], self.cy[1]),
                              size=(self.size[0], 1),
                              side='back', bc=self.bc[3],
                              sid=self.sid, inner=self.inner)

        self.faces = (self.face_left, self.face_right,
                      self.face_front, self.face_back)

    def __iter__(self):
        return iter(self.faces)

    def __fargs__(self):
        return f"origin={self.origin}, size={self.size}, bc={self.bc}, sid={self.sid}"


class Obstacle(Cuboid):
    pass


class Domain(Cuboid):
    pass


class Face(Box):

    _sides = {'right': (0, 1), 'left': (0, -1),    # (axis, normal)
              'back': (1, 1), 'front': (1, -1),
              'top': (2, 1), 'bottom': (2, -1)}

    _opposites = {'right': 'left', 'left': 'right',
                  'back': 'front', 'front': 'back',
                  'top': 'bottom', 'bottom': 'top'}

    _attributes = ['clamped', 'bounded', 'free', 'colinear', 'overlapped', 'covered']

    def __init__(self, origin, size, bc, side, sid, env=None, inner=False):
        super().__init__(origin, size, bc=bc, env=env, inner=inner)

        self.sid = sid
        self.side = side
        self.opposite = self._opposites[side]
        self.axis, self.normal = self._sides[self.side]
        self.not_axis = tuple(set(range(self.ndim)).difference((self.axis, )))
        self.loc = self.cn[self.axis][0]
        if inner:
            self.normal = - self.normal

        self.clamped = False
        self.bounded = False
        self.overlapped = False
        self.covered = False
        self.colinear = False
        self.update_indices_u()

    def update_indices_u(self):
        """ Update indices of the uncentered areas."""
        if self.clamped and not self.inner:
            self.indices_u = ()
        else:
            self.indices_u = self.box().indices

    @property
    def free(self):
        return all(not getattr(self, attr) for attr in self._attributes if attr != 'free')

    @property
    def description(self):
        return '/'.join([c for c in self._attributes if getattr(self, c)])

    def _check_args(self):
        """ Check input arguments.
        Note: Overload _check_args to impose bc to be 1-character string."""

        if any(len(self.size) != len(s) for s in [self.origin, self.env]):
            raise ValueError('origin, size, and env must have the same dimension')

        if len(self.bc) != 1:
            raise ValueError('bc must be a 1-character str')

        if any(s == 0 for s in self.size):
            raise ValueError('Size of the object must be at least 1')

    def box(self, N=5):
        """ Return a box extending N points ahead of the object."""

        if self.normal == -1:
            origin = tuple(max(0, r.start - N + 1) if self.axis == i else r.start
                           for i, r in enumerate(self.rn))
        else:
            origin = self.origin

        size = tuple(N if i == self.axis else s for i, s in enumerate(self.size))
        bc = ['X'] * len(self.origin) * 2
        bc[2*self.axis + max(0, self.normal)] = self.bc

        return Box(origin=origin, size=size, bc=''.join(bc), env=self.env)

    def __fargs__(self):

        s = f"origin={self.origin}, size={self.size}, "
        s += f"bc={self.bc}, side={self.side}, sid={self.sid}, [{self.description}]"

        return s


class BoxSet:

    def __init__(self, shape, bc, subs=None, stencil=11):

        self.shape = shape
        self.bc = bc
        self.subs = subs
        self.stencil = stencil

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """ Custom __str__ following dataclass style. """
        if getattr(self, '__fargs__', None):
            return f"{type(self).__name__}({self.__fargs__()})"

        return f"{type(self).__name__}"

    def __fargs__(self):
        s = f'shape={self.shape} -- {len(self)} elements:'
        for edge in self.subs:
            s += f'\n\t- {edge}'
        return s


class ObstacleSet(BoxSet):
    """ Collection of Obstacle objects. """

    def __init__(self, shape, bc, subs=None, stencil=11):

        super().__init__(shape, bc=bc, subs=subs, stencil=stencil)

        self.faces = tuple(_it.chain(*[sub.faces for sub in subs]))
        self.bounds = Obstacle(origin=(0, ) * len(shape), size=shape, bc=self.bc, env=shape, inner=True).faces
        self.obstacles = {o.sid:o for o in self.subs}

        # Update enveloppe (global domain) of each obstacle
        for obs in self.subs:
            obs.env = self.shape

        self.update_face_description()

    @property
    def faces_vs_subs(self):
        return [(f, o) for f, o in _it.product(self.faces, self.subs) if f.sid != o.sid]

    def update_face_description(self):

        # Faces not clamped to global domain
        for f in self.faces:
            if f.loc in [0, self.shape[f.axis] - 1]:
                f.clamped = True

        # Face is fully covered by an obtacle
        self.covered = tuple((f, o) for f, o in self.faces_vs_subs if f.issubset(o))
        for f in self.faces:
            if f in _it.chain(*self.covered):
                f.covered = True

        # Faces bounded by global domain
        for f in self.faces:
            bounded = [(r.start == 0, r.stop == s) for i, (r, s) in enumerate(zip(f.rn, self.shape))
                        if i != f.axis]
            if any(_it.chain(*bounded)):
                f.bounded = True

        self.bounded = tuple(f for f in self.faces if f.bounded)

        # Overlapped or covered by another obstacle, and "colinear"
        self.overlapped = tuple((f, o) for f, o in self.faces_vs_subs
                                 if not f.clamped and not f.covered and f.intersects(o))
        self.colinear = tuple((f, o) for f, o in self.overlapped
                               if f.intersects(o.faces[2*f.axis + max(0, f.normal)]))
        self.overlapped = tuple(it for it in self.overlapped if it not in self.colinear)

        for f in self.faces:
            if f in _it.chain(*self.overlapped):
                f.overlapped = True
            if f in _it.chain(*self.colinear):
                f.colinear = True

        # Overlapped bounds
        self.overlapped_bounds = tuple((f, o) for f, o in _it.product(self.bounds, self.subs)
                                        if f.intersects(o))

        self.free = tuple(f for f in self.faces if f.free)
        self.not_clamped = tuple(f for f in self.faces if not f.clamped and not f.covered)

        # Combination of all faces
        combs = _it.combinations(self.bounds + self.not_clamped, r=2)
        self.face_combination = tuple((f1, f2) for f1, f2 in combs
                                       if f1.sid != f2.sid and f1.axis == f2.axis)




if __name__ == "__main__":
    pass