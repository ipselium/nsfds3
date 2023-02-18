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
# Creation Date : 2022-06-17 - 15:22:15
"""
-----------

Module `mesh` provides three classes to build meshes:

    * :py:function:`build`: Factory function to build a mesh
    * :py:class:`RegularMesh`: Build regular cartesian mesh
    * :py:class:`AdaptativeMesh`: Build adaptative cartesian mesh
    * :py:class:`CurvilinearMesh`: Build curvilinear mesh

-----------
"""

import re as _re
import numpy as _np
from ._cdomain import ComputationDomains
from ._geometry import ObstacleSet
import nsfds3.graphics as _graphics


class GridError(Exception):
    """ Exception when wrong grid parameters. """


class RegularMesh:
    """ Build cartesian regular grid

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with 2 or 3 int objects.
    steps : tuple, optional
        Spatial steps. Must be a tuple with 2 or 3 float objects.
    origin : tuple, optional
        Origin of the grid. Must be a tuple with 2 or 3 int objects.
    bc : {'[ARZPW][ARZPW][ARZPW][ARZPW][[ARZPW][ARZPX]]'}, optional
        Boundary conditions. Must be a 4 characters string.
    obstacles : :py:class:`fdgrid.domains.Domain`, optional
        Obstacles in the computation domain.
    Npml : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    stencil : int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).
    flat : {(axe, index)}, optional
        Flat version of the mesh. The cut is made along axe at index.

    See also
    --------
    :py:class:`AdaptativeMesh`,
    :py:class:`CurvilinearMesh`,
    :py:mod:`fdgrid.templates`

    """

    def __init__(self, shape, steps=None, origin=None, bc=None, obstacles=None,
                 Npml=15, stencil=11, flat=None):

        self.shape = shape
        self.steps = steps
        self.origin = origin
        self.stencil = stencil
        self.Npml = Npml
        self.obstacles = obstacles
        self.bc = bc
        self.flat = flat

        self._check_arguments_dims()
        self._check_volume()

        self._set_attributes(('nx', 'ny', 'nz'), self.shape)
        self._set_attributes(('dx', 'dy', 'dz'), self.steps)
        self._set_attributes(('odx', 'ody', 'odz'), 1 / _np.array(self.steps))
        self._set_attributes(('ix0', 'iy0', 'iz0'), self.origin)
        self._update_arguments()

        self.obstacles = ObstacleSet(self.shape, self.obstacles, stencil=self.stencil)

        self._check_bc()
        self._check_grid()

        self._make_grid()
        self._find_subdomains()

    def _axes(self, values=range(3)):
        return [n for i, n in enumerate(values) if i != self.flat_ax]

    def _set_attributes(self, names, values):
        """ Helper method to set attributes. """
        values = self._axes(values)
        _ = [setattr(self, attr, val) for attr, val in zip(names, values)]

    def _check_arguments_dims(self):
        """ Check input arguments of the mesh. """

        if len(self.shape) not in [2, 3]:
            raise ValueError('Shape must be or dim 2 or 3')

        if not self.origin:
            self.origin = (0, ) * len(self.shape)

        if not self.steps:
            self.steps = (1, ) * len(self.shape)

        if len(self.shape) != len(self.steps) or len(self.shape) != len(self.origin):
            raise ValueError('shape, steps, origin must have coherent dim.')

        if not self.bc:
            self.bc = 'W' * len(self.shape) * 2
        elif len(self.bc) not in [4, 6]:
            raise ValueError('bc must be of dim 4 or 6')
        else:
            self.bc = self.bc.upper()

        if not self.obstacles:
            self.obstacles = []

    def _update_arguments(self):

        self.shape = tuple(getattr(self, attr, None) for attr
                           in ('nx', 'ny', 'nz') if getattr(self, attr, None) is not None)
        self.origin = tuple(getattr(self, attr, None) for attr
                            in ('ix0', 'iy0', 'iz0') if getattr(self, attr, None) is not None)
        self.steps = tuple(getattr(self, attr, None) for attr
                           in ('dx', 'dy', 'dz') if getattr(self, attr, None) is not None)
        if self.flat and len(self.bc) == 6:
            self.bc = ''.join(bc for i, bc in enumerate(self.bc)
                              if i not in [self.flat_ax, self.flat_ax + 1])

    def _check_volume(self):
        """ Check volume of the mesh. """
        if self.flat:
            self.flat_ax, self.flat_idx = self.flat
            self.flat_plane = self._axes()
            if self.flat_ax not in range(3):
                raise ValueError('flat[0] (axis) must be 0, 1, or 2')
            if self.flat_idx not in range(self.shape[self.flat_ax]):
                raise ValueError('flat[0] (index) must be in the domain')
        else:
            self.flat_ax, self.flat_idx = None, None

        if self.obstacles and self.flat:
            self.obstacles = [obs.flatten(self.flat_ax) for obs in self.obstacles
                              if self.flat_idx in obs.ranges[self.flat_ax]]

    @property
    def volumic(self):
        """ Return True if mesh is 3d. """
        return not self.flat and len(self.shape) == 3

    def _check_bc(self):

        regex = [r'[^P].P...', r'P.[^P]...', r'.[^P].P..', r'.P.[^P]..',
                 r'..[^P].P.', r'..P.[^P].', r'...[^P].P', r'...P.[^P]']

        if not _re.match(r'^[ZRAPW]*$', self.bc):
            raise ValueError("bc must be combination of 'ZRAPW'!")

        if any(_re.match(r, self.bc) for r in regex):
            msg = "periodic condition must be on both sides of the domain,"
            msg += " i.e. '(P.P...)'|'(.P.P..)'"
            raise ValueError(msg)

    def _check_grid(self):

        if any(s > _np.iinfo(_np.int16).max for s in self.shape):
            raise GridError(f'At least 1 dimension of the mesh exceeds {_np.iinfo(_np.int16)}')

        if self.Npml < self.stencil:
            raise GridError("Number of points of PML must be larger than stencil")

        if any(i0 >= N for i0, N in zip(self.origin, self.shape)):
            raise GridError("Origin of the domain must be in the domain")

    def _make_grid(self):
        """ Make grid. """
        self.x = (_np.arange(self.nx) - self.ix0) * self.dx
        self.y = (_np.arange(self.ny) - self.iy0) * self.dy
        if self.volumic:
            self.z = (_np.arange(self.nz) - self.iz0) * self.dz
            self.axis = (self.x, self.y, self.z)
        else:
            self.axis = (self.x, self.y)

    def _find_subdomains(self):
        """ Divide the computation domain into subdomains. """

        self.compdom = ComputationDomains(self.shape, self.obstacles,
                                          self.bc, self.stencil, self.Npml)

        self.domains = self.compdom.domains
        self.udomains = [sub for sub in self.domains if sub.tag in [(0, 0), (0, 0, 0)]]

    def get_obstacles(self):
        """ Get obstacles coordinates. """
        return [o.coords for o in self.obstacles]

    def show(self, dpi=800, obstacles=True, domains=False, bounds=True, only_mesh=False):
        """ Plot grid.

        todo :
            - BC profiles, figsize, pml, probes, filename
            - Take one division over N(=4)
            - evolution of the (dx, dy, dz) steps
        """

        viewer = _graphics.CDViewer(self)
        viewer.show(dpi=dpi, obstacles=obstacles, domains=domains, bounds=bounds,
                    only_mesh=only_mesh)

    def __str__(self):
        s = f"Cartesian {'x'.join(str(n) for n in self.shape)} points grid "
        s += f'with {self.bc} boundary conditions:\n\n'
        s += f"\t* Spatial step : ({', '.join(str(n) for n in self.steps)})\n"
        s += f"\t* Origin       : ({', '.join(str(n) for n in self.origin)})\n"
        if 'A' in self.bc:
            s += f'\t* Points in PML: {self.Npml}\n'
        s += f'\t* Max stencil  : {self.stencil}\n'

        return s

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":

    from .templates import TestCases

    shape = 64, 96, 48
    steps = 1e-4, 1e-4, 1e-4
    origin = 32, 32, 32
    stencil = 3

    test_cases = TestCases(shape, stencil)
    mesh = RegularMesh(shape, steps, origin, obstacles=test_cases.case9, bc='WWWWWW', stencil=3)
    mesh.show(dpi=900)
