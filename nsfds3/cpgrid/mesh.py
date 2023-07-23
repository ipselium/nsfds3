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
The `mesh` module provides tools to build meshes:

    * :py:func:`build_mesh`: Factory function to build a mesh grid from a given configuration
    * :py:class:`CartesianGrid`: Build cartesian grid
    * :py:class:`CurvilinearGrid`: Build curvilinear grid
"""

import re
import numpy as np
from rich.console import Console
from nsfds3.cpgrid.cdomain import ComputationDomains
from nsfds3.cpgrid.geometry import ObstacleSet
from nsfds3.cpgrid import utils
import nsfds3.graphics as graphics

from libfds.cmaths import curvilinear2d_trans, curvilinear3d_trans
from libfds.cmaths import curvilinear2d_metrics, curvilinear3d_metrics


console = Console()


def build_mesh(cfg):
    """ Return Grid from :py:class:`nsfds3.solver.CfgSetup` configuration. """
    if getattr(cfg, 'curvfunc', None):
        return CurvilinearGrid.from_cfg(cfg)
    return CartesianGrid.from_cfg(cfg)


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
    obstacles : list of :py:class:`nsfds3.cpgrid.Obstacle`, optional
        List of :py:class:`nsfds3.cpgrid.Obstacle` objects in the computation domain.
    bz_n : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    bz_stretch_factor : float, optional
        Factor reach at the end of the stretching zone
    bz_stretch_order : float, optional
        Order of the streching function
    stencil : int, optional
        Size of the finite difference stencil [obsolete].

    Note
    ----
    One can override make_grid() method to customize (x, y, z)

    See also
    --------
    :py:func:`nsfds3.cpgrid.build_mesh`
    :py:class:`nsfds3.cpgrid.CurvilinearGrid`
    :py:class:`nsfds3.cpgrid.TestCases`
    """

    mesh_type = "Cartesian"

    def __init__(self, shape, steps=None, origin=None, bc=None, obstacles=None,
                 bz_n=20, bz_stretch_factor=2, bz_stretch_order=3, stencil=11, free=True):

        self.obstacles = [] if obstacles is None else obstacles
        self.bz_n = bz_n
        self.bz_stretch_factor = bz_stretch_factor
        self.bz_stretch_order = bz_stretch_order
        self.stencil = stencil
        self.free = free
        self.shape, self.steps, self.origin, self.bc = utils.parse_grid_parameters(shape,
                                                                                   steps,
                                                                                   origin,
                                                                                   bc,
                                                                                   bz_n)

        self.ndim = len(shape)
        self._set_attributes(('nx', 'ny', 'nz'), self.shape)
        self._set_attributes(('dx', 'dy', 'dz'), self.steps)

        self.obstacles = ObstacleSet(self.shape, self.bc, self.obstacles, stencil=self.stencil)

        self.make_grid()
        self._set_axis_flags()
        self.find_subdomains()

        def bounds(i, ax, bound):
            b1 = [bound[0] if i == s else slice(None) for s in range(len(ax.shape))]
            b2 = [bound[1] if i == s else slice(None) for s in range(len(ax.shape))]
            return ax[tuple(b1)].min(), ax[tuple(b2)].max()

        self.domain_limits = [(axe.min(), axe.max()) for axe in self.paxis]
        self.buffer_limits = [bounds(i, ax, bound) for i, (ax, bound) in
                              enumerate(zip(self.paxis, utils.buffer_bounds(self.bc, self.bz_n)))]

    @classmethod
    def from_cfg(cls, cfg):
        """ Build grid from configuration. """
        args, kwargs = cfg.get_grid_configuration()
        return cls(*args, **kwargs)

    def _set_attributes(self, names, values):
        """ Helper method to set attributes. """
        _ = [setattr(self, attr, val) for attr, val in zip(names, values)]

    def _set_axis_flags(self):
        """ Set flag to specify if axis has regular (s) or irregular (v) spacing. """
        self.flag_x = 's' if np.allclose(np.diff(self.x), self.dx) else 'v'
        self.flag_y = 's' if np.allclose(np.diff(self.y), self.dy) else 'v'
        if self.ndim == 3:
            self.flag_z = 's' if np.allclose(np.diff(self.z), self.dz) else 'v'

    def find_subdomains(self):
        """ Divide the computation domain into subdomains. """

        self._computation_domains = ComputationDomains(self.shape, self.obstacles,
                                                       self.bc, self.bz_n, self.stencil,
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
        """ Return a string specifying the axis that are not regularly spaced. """
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
        stretch = 1 + max(self.bz_stretch_factor - 1, 0)  * np.linspace(0, 1, self.bz_n) ** self.bz_stretch_order

        self.x = np.arange(self.nx, dtype=float) - int(self.nx/2)
        self.y = np.arange(self.ny, dtype=float) - int(self.ny/2)

        if self.bc[0] == 'A':
            self.x[:self.bz_n] *= stretch[::-1]
        if self.bc[1] == 'A':
            self.x[-self.bz_n:] *= stretch

        if self.bc[2] == 'A':
            self.y[:self.bz_n] *= stretch[::-1]
        if self.bc[3] == 'A':
            self.y[-self.bz_n:] *= stretch

        self.x *= self.dx
        self.y *= self.dy

        self.x -= self.x[self.origin[0]]
        self.y -= self.y[self.origin[1]]

        if self.ndim == 3:
            self.z = np.arange(self.nz, dtype=float) - int(self.nz/2)
            if self.bc[4] == 'A':
                self.z[:self.bz_n] *= stretch[::-1]
            if self.bc[5] == 'A':
                self.z[-self.bz_n:] *= stretch
            self.z *= self.dz
            self.z -= self.z[self.origin[2]]

    def show(self, **kwargs):
        """ Plot grid.

        todo :
            - BC profiles
        """
        viewer = graphics.MeshViewer(self)
        viewer.show(**kwargs)

    def __str__(self):

        s = f"* {type(self).__name__} : \n"
        s += f"\t- Grid                : {'x'.join(str(n) for n in self.shape)} points grid\n"
        s += f'\t- boundary conditions : {self.bc}\n'
        s += f"\t- Spatial step        : ({', '.join(str(n) for n in self.steps)})\n"
        s += f"\t- Origin              : ({', '.join(str(n) for n in self.origin)})\n"
        if 'A' in self.bc:
            s += '\t* Buffer zone  :\n'
            s += f'\t\t* Number of points : {self.bz_n}\n'
            s += f'\t\t* Stretch factor   : {self.bz_stretch_factor}\n'
            s += f'\t\t* Stretch order    : {self.bz_stretch_order}\n'
            s += f'\t\t* Stretched axis   : {self.stretched_axis}\n\n'
        if self.obstacles:
            s += f"\t- Obstacles           : {self.obstacles}\n"
        if hasattr(self, 'curvfunc'):
            s += f"\t- Curvilinear         : {self.curvfunc}\n"
        return s

    def __repr__(self):
        return self.__str__()


class CurvilinearGrid(CartesianGrid):
    """ Build curvilinear grid

    Parameters
    ----------
    shape: tuple
        Size of the domain. Must be a tuple with 2 or 3 int objects.
    steps: tuple, optional
        Spatial steps. Must be a tuple with 2 or 3 float objects.
    origin: tuple, optional
        Origin of the grid. Must be a tuple with 2 or 3 int objects.
    bc: {'[APW][APW][APW][APW][[APW][APW]]'}, optional
        Boundary conditions. Must be a 4 or 6 characters string corresponding to
        left, right, front, back, bottom, and top boundaries, respectively.
        A stands for non reflecting boundary, W for non slip condition, and P for
        periodic boundary.
    obstacles : list of :py:class:`nsfds3.cpgrid.Obstacle`, optional
        List of :py:class:`nsfds3.cpgrid.Obstacle` objects in the computation domain.
    curvfunc: func
        Function to operate curvilinear transformation
    bz_n: int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    bz_stretch_factor: float, optional
        Factor reach at the end of the stretching zone
    bz_stretch_order: float, optional
        Order of the streching function
    stencil: int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).

    See also
    --------
    :py:func:`nsfds3.cpgrid.build_mesh`
    :py:class:`nsfds3.cpgrid.CurvilinearGrid`
    :py:class:`nsfds3.cpgrid.TestCases`
    """

    mesh_type = "Curvilinear"

    def __init__(self, shape, steps=None, origin=None, bc=None, obstacles=None,
                 curvfunc=None, bz_n=20,
                 bz_stretch_factor=2, bz_stretch_order=3, stencil=11, free=True):

        if curvfunc:
            self.curvfunc = curvfunc
        else:
            self.curvfunc = self._curvfunc

        super().__init__(shape, steps=steps, origin=origin, bc=bc, obstacles=obstacles,
                         bz_n=bz_n, bz_stretch_factor=bz_stretch_factor, bz_stretch_order=bz_stretch_order,
                         stencil=stencil, free=free)

        self.check_metrics()

    @staticmethod
    def _curvfunc(*args):
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
            self.xp, self.yp, self.zp = self.curvfunc(x, y, z)
            J = curvilinear3d_trans(self.xp, self.yp, self.zp, x, y, z)
            J = [np.array(v) for v in J]
            self.J, self.dx_du, self.dx_dv, self.dx_dw, self.dy_du, self.dy_dv, self.dy_dw, self.dz_du, self.dz_dv, self.dz_dw = J
        else:
            x, y = np.meshgrid(self.x, self.y, indexing='ij')
            self.xp, self.yp = self.curvfunc(x, y)
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
        del attributes['curvfunc']
        return attributes


if __name__ == "__main__":

    from nsfds3.cpgrid import TestCases

    shape = 256, 128
    steps = 1e-4, 1e-4
    origin = 32, 32
    bc = 'WWWW'

    obstacles = TestCases.single(shape)
    mesh = CartesianGrid(shape, steps=steps, origin=origin, obstacles=obstacles, bc=bc)
    mesh.show(dpi=900)
