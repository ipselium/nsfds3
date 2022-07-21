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
# Creation Date : 2022-06-09 - 22:15:26
"""
-----------

Some helper classes and functions to represent meshes graphically.

-----------
"""

import numpy as _np
import plotly.graph_objects as _go
from plotly.subplots import make_subplots


def fig_scale(ax1, ax2, ref=None):
    """ Return ideal size ratio. """
    if isinstance(ax1, (tuple, list)):
        s1 = sum(ax.max() - ax.min() for ax in ax1)
    else:
        s1 = ax1.max() - ax1.min()

    if isinstance(ax2, (tuple, list)):
        s2 = sum(ax.max() - ax.min() for ax in ax2)
    else:
        s2 = ax2.max() - ax2.min()

    ratio = min(s1, s2) / max(s1, s2)
    b1 = 1 / (1 + ratio)
    b2 = 1 - b1

    if ref:
        if s1 < s2:
            b2, b1 = ref, ref * b1 / b2
        else:
            b1, b2 = ref, ref * b2 / b1

    return b2 if s1 < s2 else b1, b2 if s2 < s1 else b1


class CDViewer:
    """ Computation domain viewer. """

    def __init__(self, obj):
        self.shape = obj.shape
        self.volumic = obj.volumic
        self.stencil = obj.stencil
        self.obstacles = obj.obstacles
        self.domains = obj.domains
        self.udomains = [s for s in self.domains if s.tag != (0, ) * len(self.shape)]
        self.traces = []

        self.x = getattr(obj, 'x', _np.arange(self.shape[0]))
        self.y = getattr(obj, 'y', _np.arange(self.shape[1]))
        if len(self.shape) == 3:
            self.z = getattr(obj, 'z', _np.arange(self.shape[1]))

    @staticmethod
    def _grid_traces(ax1, ax2):
        """ Make grid traces. """
        traces = []
        kwargs = dict(mode='lines', line={'color': 'rgba(0,0,0,0.1)'})

        x = _np.ones_like(ax2)
        for ix in ax1:
            traces.append(_go.Scatter(x=ix * x, y=ax2, **kwargs))

        y = _np.ones_like(ax1)
        for iy in ax2:
            traces.append(_go.Scatter(x=ax1, y=iy * y, **kwargs))

        return traces

    def _object_traces(self, ax1, ax2, side=None, reverse=None, kind='obstacle', bounds=False):
        """ Make obstacles traces. """
        traces = []

        if kind == 'domains':
            if bounds:
                obj = self.domains
            else:
                obj = self.domains.inner_objects
            kwargs = {'fill': "toself", 'fillcolor': 'rgba(0.39, 0.98, 0.75, 0.1)',
                      'line': {'color': 'rgba(0.39, 0,98, 0.75)'}}
        else:
            obj = self.obstacles
            kwargs = {'fill': "toself", 'fillcolor': 'rgba(0, 0, 0, 0.1)',
                      'fillpattern': {'shape': 'x'},
                      'line': {'color': 'rgba(0, 0, 0)'}}

        if not self.volumic:
            for obs in obj:
                ix, iy = obs.vertices
                traces.append(_go.Scatter(x=ax1[ix, ], y=ax2[iy, ],
                                          name=f'obs{obs.sid}',
                                          **kwargs
                                          ))
        elif side:
            for face in getattr(obj.faces, side):
                ix, iy = face.vertices[face.not_axis, :]
                if reverse == 'reversed':
                    ix, iy = iy, ix
                traces.append(_go.Scatter(x=ax1[ix], y=ax2[iy], name=f'obs{face.sid}', **kwargs))

        return traces

    def show(self, dpi=800, obstacles=True, domains=False, bounds=True, only_mesh=False):
        """ Plot grid.

        todo :
            - BC profiles, figsize, pml, probes, filename
            - Take one division over N(=4)
            - evolution of the (dx, dy, dz) steps
        """
        if self.volumic:
            fig = self._grid3d(obstacles=obstacles, domains=domains, bounds=bounds,
                               only_mesh=only_mesh)
            width, height = fig_scale((self.x, self.z), (self.y, self.z), ref=dpi)
        else:
            fig = self._grid2d(obstacles=obstacles, domains=domains, bounds=bounds)
            width, height = fig_scale(self.x, self.y, ref=dpi)

        fig.update_layout(showlegend=False, height=height, width=width,
                          plot_bgcolor='rgba(0,0,0,0)')
        fig.show()

    def _grid2d(self, obstacles=True, domains=False, bounds=False):
        """ Show 2d grid. """
        fig = _go.Figure()
        fig.add_traces(self._grid_traces(self.x, self.y))
        if self.obstacles and obstacles:
            fig.add_traces(self._object_traces(self.x, self.y, kind='obstacles'))
        if domains:
            fig.add_traces(self._object_traces(self.x, self.y, kind='domains', bounds=bounds))
        fig.update_xaxes(title=r'$x \, [m]$',
                         autorange=False, automargin=True,
                         range=[self.x.min(), self.x.max()],
                         )
        fig.update_yaxes(title=r'$y \, [m]$',
                         scaleanchor="x",
                         autorange=False, automargin=True,
                         range=[self.y.min(), self.y.max()],
                         )
        return fig

    def _grid3d(self, obstacles=True, domains=False, bounds=False, only_mesh=False):
        """ Show 3d grid. """
        # Figure
        if only_mesh:
            fig = _go.Figure()
            fig.add_traces(self._mesh3d(obstacles=True, domains=False, bounds=True))
        else:
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
                       ((r'', self.x),
                        (r'$z \, [m]$', self.z),
                        ('$x \, [m]$', self.x)),
                       ((r'$y \, [m]$', self.y),
                        (r'', self.y),
                        ('$z \, [m]$', self.z)))

            for row, col, side, rev, anchor, (ax1l, ax1), (ax2l, ax2) in axis:

                fig.add_traces(self._grid_traces(ax1, ax2), rows=row, cols=col)
                if self.obstacles and obstacles:
                    fig.add_traces(self._object_traces(ax1, ax2, side, rev,
                                   kind='obstacles'),
                                   rows=row, cols=col)
                if domains:
                    fig.add_traces(self._object_traces(ax1, ax2, side, rev,
                                   kind='domains', bounds=bounds),
                                   rows=row, cols=col)

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

            fig.add_traces(self._mesh3d(obstacles=True, domains=False, bounds=True),
                           rows=2, cols=2)

        return fig

    def _mesh3d(self, obstacles=True, domains=False, bounds=True):

        data = []

        if obstacles:
            for sub in self.obstacles:
                data.append(_go.Mesh3d(x=sub.vertices[0],
                                       y=sub.vertices[1],
                                       z=sub.vertices[2],
                                       intensity=_np.linspace(0, 1, 8, endpoint=True),
                                       name='y',
                                       opacity=1,
                                       alphahull=0,
                                       showscale=False,
                                       flatshading=True   # to hide the triangles
                                       ))

        if domains:
            if bounds:
                domains = self.domains
            else:
                domains = self.domains.inner_objects

            for sub in domains:
                data.append(_go.Mesh3d(x=sub.vertices[0],
                                       y=sub.vertices[1],
                                       z=sub.vertices[2],
                                       hovertext=str(sub.tag),
                                       #colorscale=[[0, 'gold'],
                                       #            [0.5, 'mediumturquoise'],
                                       #            [1, 'magenta']],
                                       intensity=_np.linspace(0, 1, 8, endpoint=True),
                                       # i, j and k give the vertices of triangles
                                       #i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                                       #j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                                       #k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                                       opacity=0.5,
                                       alphahull=1,
                                       showscale=True,
                                       flatshading=True   # to hide the triangles
                                       ))
        return data
