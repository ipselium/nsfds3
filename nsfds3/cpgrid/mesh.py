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

    * :py:function:`build`: Factory function to build a mesh grid
    * :py:class:`CartesianGrid`: Build cartesian grid
    * :py:class:`CurvilinearGrid`: Build curvilinear grid

-----------
"""

import re
import numpy as np
from rich.console import Console
from .cdomain import ComputationDomains
from .geometry import ObstacleSet
from .utils import buffer_bounds
import nsfds3.graphics as graphics

from libfds.cmaths import curvilinear2d_trans, curvilinear3d_trans
from libfds.cmaths import curvilinear2d_metrics, curvilinear3d_metrics


console = Console()


def build_mesh(cfg):
    if getattr(cfg, 'curvilinear_func', None):
        return CurvilinearGrid.from_cfg(cfg)
    return CartesianGrid.from_cfg(cfg)


class GridError(Exception):
    """ Exception raised when grid parameters are wrong. """


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

    mesh_type = "Cartesian"

    def __init__(self, shape, steps=None, origin=None, bc=None, obstacles=None,
                 nbz=20, stretch_factor=2, stretch_order=3, stencil=11, free=True):

        self.shape = shape
        self.steps = steps
        self.origin = origin
        self.ndim = len(shape)
        self.bc = bc
        self.obstacles = obstacles
        self.nbz = nbz
        self.stretch_factor = stretch_factor
        self.stretch_order = stretch_order
        self.stencil = stencil
        self.free = free

        self.check_arguments_dims()
        self.set_attributes(('nx', 'ny', 'nz'), self.shape)
        self.set_attributes(('dx', 'dy', 'dz'), self.steps)

        self.obstacles = ObstacleSet(self.shape, self.bc, self.obstacles, stencil=self.stencil)

        self.check_bc()
        self.check_grid()
        self.make_grid()
        self.set_axis_flags()
        self.find_subdomains()

        def bounds(i, ax, bound):
            b1 = [bound[0] if i == s else slice(None) for s in range(len(ax.shape))]
            b2 = [bound[1] if i == s else slice(None) for s in range(len(ax.shape))]
            return ax[tuple(b1)].min(), ax[tuple(b2)].max()

        self.domain_limits = [(axe.min(), axe.max()) for axe in self.paxis]
        self.buffer_limits = [bounds(i, ax, bound) for i, (ax, bound) in enumerate(zip(self.paxis, buffer_bounds(self.bc, self.nbz)))]

    @classmethod
    def from_cfg(cls, cfg):
        args, kwargs = cfg.get_mesh_config()
        return cls(*args, **kwargs)

    def check_arguments_dims(self):
        """ Check dimensions of input arguments. """

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

    def set_attributes(self, names, values):
        """ Helper method to set attributes. """
        _ = [setattr(self, attr, val) for attr, val in zip(names, values)]

    def check_bc(self):
        """ Check that boundary condition are well declared. """
        regex = [r'[^P]P..', r'P[^P]..', r'[^P]P....', r'P[^P]....',
                 r'..[^P]P', r'..P[^P]', r'..[^P]P..', r'..P[^P]..',
                 r'....[^P]P', r'....P[^P]',]

        if not re.match(r'^[APW]*$', self.bc):
            raise ValueError(f"bc must be combination of {2 * len(self.shape)} chars among 'APW'!")

        if any(re.match(r, self.bc) for r in regex):
            msg = "periodic condition must be on both sides of the domain,"
            msg += " i.e. '(PP....)'|'(..PP..)'|'(....PP)'"
            raise ValueError(msg)

        if not all([n - self.nbz * (self.bc[2*i:2*i + 2].count('A')) > 11 for i, n in enumerate(self.shape)]):
            raise GridError('One of the dimension is too small to setup a buffer zone.')

    def check_grid(self):
        """ Check that grid is well declared. """
        if any(s > np.iinfo(np.int16).max for s in self.shape):
            raise GridError(f'At least 1 dimension of the mesh exceeds {np.iinfo(np.int16)}')

        if any(i0 >= N for i0, N in zip(self.origin, self.shape)):
            raise GridError("Origin of the domain must be in the domain")

    def set_axis_flags(self):
        """ Set flag to specify if axis has regular (s) or irregular (v) spacing. """
        self.flag_x = 's' if np.allclose(np.diff(self.x), self.dx) else 'v'
        self.flag_y = 's' if np.allclose(np.diff(self.y), self.dy) else 'v'
        if self.ndim == 3:
            self.flag_z = 's' if np.allclose(np.diff(self.z), self.dz) else 'v'

    def find_subdomains(self):
        """ Divide the computation domain into subdomains. """

        self._computation_domains = ComputationDomains(self.shape, self.obstacles,
                                                       self.bc, self.nbz, self.stencil, 
                                                       free=self.free)

        self.bounds = self._computation_domains.bounds
        self.buffer = self._computation_domains.buffer
        self.cdomains = self._computation_domains.cdomains
        self.xdomains = self._computation_domains.xdomains
        self.ydomains = self._computation_domains.ydomains
        if self.ndim == 3:
            self.zdomains = self._computation_domains.zdomains

    @property
    def stretched_axis(self):
        """ Return a string specifying the axis that are not regular. """
        s = ''
        if self.flag_x == 'v':
            s += 'x'
        if self.flag_y == 'v':
            s += 'y'
        if self.ndim == 3:
            if self.flag_z == 'v':
                s += 'z'
        return ' & '.join(list(s))

    @property
    def paxis(self):
        """ Physical axis. """
        if self.ndim == 3:
            return np.meshgrid(self.x, self.y, self.z, indexing='ij')
        return np.meshgrid(self.x, self.y, indexing='ij')

    @property
    def axis(self):
        """ Numerical axis. """
        if self.ndim == 3:
            return self.x, self.y, self.z
        return self.x, self.y

    def get_obstacles(self):
        """ Get obstacles coordinates. """
        return [o.cn for o in self.obstacles]

    def make_grid(self):
        """ Build grid. """
        stretch = 1 + max(self.stretch_factor - 1, 0)  * np.linspace(0, 1, self.nbz) ** self.stretch_order

        self.x = np.arange(self.nx, dtype=float) - int(self.nx/2)
        self.y = np.arange(self.ny, dtype=float) - int(self.ny/2)

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

        if self.ndim == 3:
            self.z = np.arange(self.nz, dtype=float) - int(self.nz/2)
            if self.bc[4] == 'A':
                self.z[:self.nbz] *= stretch[::-1]
            if self.bc[5] == 'A':
                self.z[-self.nbz:] *= stretch
            self.z *= self.dz
            self.z -= self.z[self.origin[2]]

    def show(self, backend='mpl', **kwargs):
        """ Plot grid.

        todo :
            - BC profiles
        """

        if backend == 'plotly':
            viewer = graphics.CDViewer(self)
        elif backend == 'mpl':
            viewer = graphics.MeshViewer(self)
        else:
            raise ValueError("backend must be in ('mpl', 'plotly', )")

        viewer.show(**kwargs)

    def __str__(self):
        s = f"{self.mesh_type} {'x'.join(str(n) for n in self.shape)} points grid "
        s += f'with {self.bc} boundary conditions:\n\n'
        s += f"\t* Spatial step : ({', '.join(str(n) for n in self.steps)})\n"
        s += f"\t* Origin       : ({', '.join(str(n) for n in self.origin)})\n"
        if 'A' in self.bc:
            s += '\t* Buffer zone  :\n'
            s += f'\t\t* Number of points : {self.nbz}\n'
            s += f'\t\t* Stretch factor   : {self.stretch_factor}\n'
            s += f'\t\t* Stretch order    : {self.stretch_order}\n'
            s += f'\t\t* Stretched axis   : {self.stretched_axis}\n\n'
        return s

    def __repr__(self):
        return self.__str__()


class CurvilinearGrid(CartesianGrid):
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
    curvilinear_func : func
        Function to operate curvilinear transformation
    nbz : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    stretch_factor : float, optional
        Factor reach at the end of the stretching zone
    stretch_order : float, optional
        Order of the streching function
    stencil : int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).

    See also
    --------
    :py:class:`CartesianMesh`,
    :py:mod:`cpgrid.templates`
    """

    mesh_type = "Curvilinear"

    def __init__(self, shape, steps=None, origin=None, bc=None, obstacles=None,
                 curvilinear_func=None, nbz=20,
                 stretch_factor=2, stretch_order=3, stencil=11, free=True):

        if curvilinear_func:
            self.curvilinear_func = curvilinear_func
        else:
            self.curvilinear_func = self._curvilinear_func

        super().__init__(shape, steps=steps, origin=origin, bc=bc, obstacles=obstacles,
                         nbz=nbz, stretch_factor=stretch_factor, stretch_order=stretch_order,
                         stencil=stencil, free=free)

        self.check_metrics()

    @staticmethod
    def _curvilinear_func(*args):
        return tuple([v.copy() for v in args])

    @property
    def paxis(self):
        if self.ndim == 3:
            return self.xp, self.yp, self.zp
        return self.xp, self.yp

    def make_grid(self):
        """ Make curvilinear grid.

        Note
        ----

        (x, y, z) define numerical grid
        (u, v, w) define physical grid
        """

        super().make_grid()

        # Pysical coordinates & Jacobian
        if self.ndim == 3:
            x, y, z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
            self.xp, self.yp, self.zp = self.curvilinear_func(x, y, z)
            J = curvilinear3d_trans(self.xp, self.yp, self.zp, x, y, z)
            J = [np.array(v) for v in J]
            self.J, self.dx_du, self.dx_dv, self.dx_dw, self.dy_du, self.dy_dv, self.dy_dw, self.dz_du, self.dz_dv, self.dz_dw = J
        else:
            x, y = np.meshgrid(self.x, self.y, indexing='ij')
            self.xp, self.yp = self.curvilinear_func(x, y)
            J = curvilinear2d_trans(self.xp, self.yp, x, y)
            J = [np.array(v) for v in J]
            self.J, self.dx_du, self.dx_dv, self.dy_du, self.dy_dv = J

    def check_metrics(self, rtol=1e-8):
        """ Check metrics. """
        msg = f'Warning : Metric invariants > {rtol}\n'

        if self.ndim == 3:
            invariants = curvilinear3d_metrics(self.J, self.dx_du, self.dx_dv, self.dx_dw,
                                                       self.dy_du, self.dy_dv, self.dy_dw,
                                                       self.dz_du, self.dz_dv, self.dz_dw)
        else:
            invariants = curvilinear2d_metrics(self.J, self.dx_du, self.dx_dv,
                                                 self.dy_du, self.dy_dv)

        self.invariants = [np.max(np.abs(inv[self.buffer.sn])) for inv in invariants]
        if not np.allclose(np.array(self.invariants), 0., rtol=rtol):
            inv = [f'Max {ax}-invariant {inv}\n' for ax, inv in zip(('x', 'y', 'z'), self.invariants)]
            msg += ''.join(inv)
            console.print(msg)

    def __getstate__(self):
        attributes = self.__dict__.copy()
        # can't picke external function, so delete it from instance...
        del attributes['curvilinear_func']
        return attributes


if __name__ == "__main__":

    from .templates import TestCases

    shape = 64, 96, 48
    steps = 1e-4, 1e-4, 1e-4
    origin = 32, 32, 32
    stencil = 3

    test_cases = TestCases(shape, stencil)
    mesh = CartesianGrid(shape, steps, origin, obstacles=test_cases.case9, bc='WWWWWW', stencil=3)
    mesh.show(dpi=900)
