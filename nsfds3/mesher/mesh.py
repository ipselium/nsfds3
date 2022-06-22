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
from .cdomain import ComputationDomains
from .geometry import ObstacleSet
from .graphics import fig_scale
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GridError(Exception):
    """ Exception when wrong grid parameters. """


class RegularMesh:
    """ Build cartesian regular grid

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with two int objects.
    steps : tuple
        Spatial steps. Must be a tuple with two float objects.
    origin : tuple, optional
        Origin of the grid. Must be a tuple with two int objects.
    bc : {'[ARZPW][ARZPW][ARZPW][ARZPW]'}, optional
        Boundary conditions. Must be a 4 characters string.
    obstacles : :py:class:`fdgrid.domains.Domain`, optional
        Obstacles in the computation domain.
    Npml : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    stencil : int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).

    See also
    --------
    :py:class:`AdaptativeMesh`,
    :py:class:`CurvilinearMesh`,
    :py:mod:`fdgrid.templates`

    """

    def __init__(self, shape, steps, origin=None, bc=None, obstacles=None,
                 Npml=15, stencil=11, flat=None):

        self.shape = shape
        self.steps = steps
        self.origin = origin
        self.stencil = stencil
        self.Npml = Npml
        self.obstacles = obstacles
        self.bc = bc
        self.flat = flat
        self._check_arguments()

        self.nx, self.ny, self.nz = shape
        self.dx, self.dy, self.dz = steps
        self.ix0, self.iy0, self.iz0 = origin
        self.obstacles = ObstacleSet(self.shape, self.obstacles, stencil=self.stencil)

        self._check_bc()
        self._check_grid()

        self._make_grid()
        self._find_subdomains()
        self._update_obstacles()

    def _check_arguments(self):
        """ Check input arguments of Mesh. """
        if isinstance(self.flat, (tuple, list)):
            self.flat_ax, self.flat_idx = self.flat
            if self.flat_ax not in range(3):
                raise ValueError('flat[0] (axis) must be 0, 1, or 2')
            if self.flat_idx not in range(self.shape[self.flat_ax]):
                raise ValueError('flat[0] (index) must be in the domain')

        if not self.bc:
            self.bc = 'W' * len(self.shape) * 2
        else:
            self.bc = self.bc.upper()

        if len(self.bc) != len(self.shape) * 2:
            raise ValueError('bc must define a boundary for each face of the domain.')

        if not self.obstacles:
            self.obstacles = []

        if self.obstacles and self.flat:
            self.obstacles = [obs for obs in self.obstacles
                              if self.flat_idx in obs.ranges[self.flat_ax]]

        if not self.origin:
            self.origin = (0, ) * len(self.shape)

        if len(self.shape) not in [2, 3]:
            raise ValueError('Shape must be or dim 2 or 3')

        if len(self.shape) != len(self.steps) or len(self.shape) != len(self.origin):
            raise ValueError('shape, steps, and origin must have same dim.')

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
        self.z = (_np.arange(self.nz) - self.iz0) * self.dz
        self.axis = (self.x, self.y, self.z)

    def _find_subdomains(self):
        """ Divide the computation domain into subdomains. """

        self.domain = ComputationDomains(self.shape, self.obstacles,
                                         self.bc, self.stencil, self.Npml,
                                         self.flat)

        self.udomains = self.domain.udomains
        self.cdomains = self.domain.cdomains

    def _update_obstacles(self):
        """ Update obstacles if mesh if flat. """
        if self.flat:
            obstacles = []
            for obs in self.obstacles:
                obstacles.append(obs.to_2d(axis=self.flat_ax))
            self.obstacles = obstacles

    def _make_view(self, ax1, ax2, side=None, reverse=None):
        """ Make grid and plot obstacles. """

        traces = []
        kwargs = dict(mode='lines', line={'color': 'rgba(0,0,0,0.1)'})

        x = _np.ones_like(ax2)
        for ix in ax1:
            traces.append(go.Scatter(x=ix * x, y=ax2, **kwargs))

        y = _np.ones_like(ax1)
        for iy in ax2:
            traces.append(go.Scatter(x=ax1, y=iy * y, **kwargs))

        kwargs = {'fill': "toself", 'fillcolor': 'rgba(0,0,0,0.1)',
                'line': {'color': 'rgba(0,0,0,0.1)'}}

        if self.flat:
            for obs in self.obstacles:
                ix, iy = obs.vertices
                traces.append(go.Scatter(x=ax1[ix, ], y=ax2[iy, ],
                                         name=f'obs{obs.sid}',
                                         **kwargs
                                         ))
        else:
            for face in getattr(self.obstacles.faces, side):
                ix, iy = face.vertices[face.not_axis, :]
                if reverse == 'reversed':
                    ix, iy = iy, ix
                traces.append(go.Scatter(x=ax1[ix], y=ax2[iy], name=f'obs{face.sid}', **kwargs))

        return traces

    def show_grid(self, dpi=800):
        """ Plot grid.

        todo :
            - BC profiles, figsize, pml, probes, filename
            - Take one division over N(=4)
            - evolution of the (dx, dy, dz) steps
        """
        if self.flat:
            fig = self._show_grid2d()
        else:
            fig = self._show_grid3d()

        width, height = fig_scale((self.x, self.z), (self.y, self.z), ref=dpi)
        fig.update_layout(showlegend=False, height=height, width=width,
                          plot_bgcolor='rgba(0,0,0,0)')
        fig.show()

    def _show_grid2d(self):
        """ Show 2d grid. """
        ax1, ax2 = [ax for i, ax in enumerate(self.axis) if i != self.flat_ax]
        fig = go.Figure(data=self._make_view(ax1, ax2))
        fig.update_xaxes(title=r'$x \, [m]$',
                         autorange=False, automargin=True,
                         range=[ax1.min(), ax1.max()],
                         )
        fig.update_yaxes(title=r'$y \, [m]$',
                         scaleanchor="x",
                         autorange=False, automargin=True,
                         range=[ax2.min(), ax2.max()],
                         )
        return fig

    def _show_grid3d(self):
        """ Show 3d grid. """
        # Figure
        fig = make_subplots(rows=2, cols=2,
                            horizontal_spacing=0.01, vertical_spacing=0.01,
                            subplot_titles=("(T)", "(R)", "(F)", ""),
                            shared_xaxes=True, shared_yaxes=True,
                            column_widths=fig_scale(self.x, self.z),
                            row_heights=fig_scale(self.y, self.z),
                            specs=[[{"type": "xy"}, {"type": "xy"}],
                                   [{"type": "xy"}, {"type": "scene"}]])

        axis = zip((1, 1, 2), (1, 2, 1),
                   ('top', 'right', 'front'),
                   (True, 'reversed', True),
                   ('x', 'x2', 'x3'),
                   ((r'', self.x), (r'$z \, [m]$', self.z), ('$x \, [m]$', self.x)),
                   ((r'$y \, [m]$', self.y), (r'', self.y), ('$z \, [m]$', self.z)))

        for row, col, side, rev, anchor, (ax1l, ax1), (ax2l, ax2) in axis:

            fig.add_traces(self._make_view(ax1, ax2, side, rev), rows=row, cols=col)

            fig.update_xaxes(row=row, col=col,
                             title=ax1l,
                             autorange=rev, automargin=True,
                             #range=[ax1.min(), ax1.max()],
                             )
            fig.update_yaxes(row=row, col=col,
                             title=ax2l,
                             scaleanchor=anchor,
                             autorange=True, automargin=True,
                             #range=[ax2.min(), ax2.max()],
                             )

        fig.add_traces(self.domain.get_traces(axis=0, obstacles=True,
                                              mask=False, domains=False,
                                              bounds=True), rows=2, cols=2)
        return fig

    def __str__(self):
        s = f'Cartesian {self.nx}x{self.ny} points grid '
        s = f'with {self.bc} boundary conditions:\n\n'
        s += f'\t* Spatial step : {(self.dx, self.dz)}\n'
        s += f'\t* Origin       : {(self.ix0, self.iz0)}\n'
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
    mesh.show_grid(dpi=900)
