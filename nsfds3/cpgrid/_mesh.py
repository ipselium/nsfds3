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
# Creation Date : 2022-06-17 - 15:22:15
"""
-----------

Module `mesh` provides three classes to build meshes:

    * :py:function:`build`: Factory function to build a mesh
    * :py:class:`CartesianGrid`: Build cartesian grid
    * :py:class:`CurvilinearGrid`: Build curvilinear grid

-----------
"""

import re as _re
import numpy as _np
from ._cdomain import ComputationDomains
from ._geometry import ObstacleSet
import nsfds3.graphics as _graphics


class GridError(Exception):
    """ Exception when wrong grid parameters. """


class CartesianGrid:
    """ Build cartesian grid

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with 2 or 3 int objects.
    steps : tuple, optional
        Spatial steps. Must be a tuple with 2 or 3 float objects.
    origin : tuple, optional
        Origin of the grid. Must be a tuple with 2 or 3 int objects.
    bc : {'[APW][APW][APW][APW][[APW][APW]]'}, optional
        Boundary conditions. Must be a 4 or 6 characters string corresponding to
        left, right, front, back, bottom, and top boundaries, respectively.
        A stands for non reflecting boundary, W for non slip condition, and P for 
        periodic boundary.
    obstacles : :py:class:`fdgrid.domains.Domain`, optional
        Obstacles in the computation domain.
    nbz : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    stretch_factor : float, optional
        Factor reach at the end of the stretching zone
    stretch_order : float, optional
        Order of the streching function
    stencil : int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).

    Note
    ----
    One can override make_grid() method to customize (x, y, z)

    See also
    --------
    :py:class:`CurvilinearMesh`,
    :py:mod:`cpgrid.templates`

    """

    def __init__(self, shape, steps=None, origin=None, bc=None, obstacles=None,
                 nbz=20, stretch_factor=2, stretch_order=3, stencil=11):

        self.shape = shape
        self.steps = steps
        self.origin = origin
        self.bc = bc
        self.obstacles = obstacles
        self.nbz = nbz
        self.stretch_factor = stretch_factor
        self.stretch_order = stretch_order
        self.stencil = stencil

        self._check_arguments_dims()
        self._set_attributes(('nx', 'ny', 'nz'), self.shape)
        self._set_attributes(('dx', 'dy', 'dz'), self.steps)

        self.obstacles = ObstacleSet(self.shape, self.obstacles, stencil=self.stencil)

        self._check_bc()
        self._check_grid()
        self.make_grid()
        self._set_axis_flags()
        self._find_subdomains()

    def _check_arguments_dims(self):
        """ Check input arguments. """

        if len(self.shape) not in [2, 3]:
            raise ValueError('Shape must be of dim 2 or 3')

        if not self.bc:
            self.bc = 'W' * len(self.shape) * 2
        elif len(self.bc) != 2 * len(self.shape):
            raise ValueError(f'Expecting bc of dim {len(2*self.shape)}')
        else:
            self.bc = self.bc.upper()

        if not self.steps:
            self.steps = (1, ) * len(self.shape)

        if not self.origin:
            self.origin = tuple([self.nbz if self.bc[2*i] == "A" else 0 for i in range(len(self.shape))])

        if len(self.shape) != len(self.steps) or len(self.shape) != len(self.origin):
            raise ValueError('shape, steps, origin must have same dims.')

        if not self.obstacles:
            self.obstacles = []

    def _set_attributes(self, names, values):
        """ Helper method to set attributes. """
        _ = [setattr(self, attr, val) for attr, val in zip(names, values)]

    def _check_bc(self):

        regex = [r'[^P]P..', r'P[^P]..', r'[^P]P....', r'P[^P]....',
                 r'..[^P]P', r'..P[^P]', r'..[^P]P..', r'..P[^P]..',
                 r'....[^P]P', r'....P[^P]',]

        if not _re.match(r'^[APW]*$', self.bc):
            raise ValueError(f"bc must be combination of {2 * len(self.shape)} chars among 'APW'!")

        if any(_re.match(r, self.bc) for r in regex):
            msg = "periodic condition must be on both sides of the domain,"
            msg += " i.e. '(PP....)'|'(..PP..)'|'(....PP)'"
            raise ValueError(msg)

        if not all([n - self.nbz * (self.bc[2*i:2*i + 2].count('A')) > 11 for i, n in enumerate(self.shape)]):
            raise GridError('One of the dimension is too small to setup a buffer zone.')

    def _check_grid(self):

        if any(s > _np.iinfo(_np.int16).max for s in self.shape):
            raise GridError(f'At least 1 dimension of the mesh exceeds {_np.iinfo(_np.int16)}')

        if any(i0 >= N for i0, N in zip(self.origin, self.shape)):
            raise GridError("Origin of the domain must be in the domain")

    def _set_axis_flags(self):
        """ Set flag to specify if axis has regular (s) or irregular (v) spacing. """
        self.flag_x = 's' if _np.allclose(_np.diff(self.x), self.dx) else 'v'
        self.flag_y = 's' if _np.allclose(_np.diff(self.y), self.dy) else 'v'
        if self.volumic:
            self.flag_z = 's' if _np.allclose(_np.diff(self.z), self.dz) else 'v'

    def _find_subdomains(self):
        """ Divide the computation domain into subdomains. """

        self._computation_domains = ComputationDomains(self.shape, self.obstacles,
                                                       self.bc, self.stencil)

        self.bounds = self._computation_domains.bounds
        self.domains = self._computation_domains.domains

    @property
    def stretched_axis(self):
        s = ''
        if self.flag_x == 'v':
            s += 'x'
        if self.flag_y == 'v':
            s += 'y'
        if self.volumic:
            if self.flag_z == 'v':
                s += 'z'
        return ' & '.join(list(s))

    @property
    def axis(self):
        if self.volumic:
            return self.x, self.y, self.z
        return self.x, self.y

    @property
    def volumic(self):
        """ Return True if mesh is 3d. """
        return len(self.shape) == 3

    def get_obstacles(self):
        """ Get obstacles coordinates. """
        return [o.coords for o in self.obstacles]

    def make_grid(self):

        stretch = 1 + max(self.stretch_factor - 1, 0)  * _np.linspace(0, 1, self.nbz) ** self.stretch_order

        self.x = _np.arange(self.nx, dtype=float) - int(self.nx/2)
        self.y = _np.arange(self.ny, dtype=float) - int(self.ny/2)

        if self.bc[0] == 'A':
            self.x[:self.nbz] *= stretch[::-1]
        if self.bc[1] == 'A':
            self.x[-self.nbz:] *= stretch

        if self.bc[2] == 'A':
            self.y[:self.nbz] *= stretch[::-1]
        if self.bc[3] == 'A':
            self.y[-self.nbz:] *= stretch

        self.x *= self.dx
        self.y *= self.dy

        self.x -= self.x[self.origin[0]]
        self.y -= self.y[self.origin[1]]

        if self.volumic:
            self.z = _np.arange(self.nz, dtype=float) - int(self.nz/2)
            if self.bc[4] == 'A':
                self.z[:self.nbz] *= stretch[::-1]
            if self.bc[5] == 'A':
                self.z[-self.nbz:] *= stretch
            self.z *= self.dz
            self.z -= self.z[self.origin[2]]

    def show(self, dpi=800, obstacles=True, domains=False, bounds=True, only_mesh=False):
        """ Plot grid.

        todo :
            - BC profiles, figsize, buffer zones, probes, filename
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
            s += '\t* Buffer zone  :\n'
            s += f'\t\t* Number of points : {self.nbz}\n'
            s += f'\t\t* Stretch factor   : {self.stretch_factor}\n'
            s += f'\t\t* Stretch order    : {self.stretch_order}\n'
            s += f'\t\t* Stretched axis   : {self.stretched_axis}'
        return s

    def __repr__(self):
        return self.__str__()


class CurvilinearGrid(CartesianGrid):
    pass


if __name__ == "__main__":

    from .templates import TestCases

    shape = 64, 96, 48
    steps = 1e-4, 1e-4, 1e-4
    origin = 32, 32, 32
    stencil = 3

    test_cases = TestCases(shape, stencil)
    mesh = CartesianGrid(shape, steps, origin, obstacles=test_cases.case9, bc='WWWWWW', stencil=3)
    mesh.show(dpi=900)
