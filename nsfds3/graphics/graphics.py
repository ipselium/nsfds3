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
import matplotlib.pyplot as _plt
import plotly.graph_objects as _go


def fig_scale(ax1, ax2, ref=None):

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

    def __init__(self, cdomain):
        self.shape = cdomain.shape
        self.stencil = cdomain.stencil
        self.obstacles = cdomain.obstacles
        self.mask = cdomain._mask
        self.domains = cdomain.domains
        self.data = []

    def show(self, axis=0, obstacles=True, mask=False, domains=False, bounds=True):
        """ Plot 3d representation of computation domain. """

        data = self.get_traces(axis=axis, obstacles=obstacles,
                               mask=mask, domains=domains, bounds=bounds)
        fig = _go.Figure(data=data)
        fig.update_layout(autosize=False,
                          width=1000,
                          height=800,)
        fig.show()

    def get_traces(self, axis=0, obstacles=True, mask=False, domains=False, bounds=True):
        """ Get traces. """

        data = []
        if obstacles:
            data.extend(self._show_obstacles())
        if domains:
            data.extend(self._show_domains(bounds))
        if mask:
            data.extend(self._show_mask(axis))

        return data

    def _show_obstacles(self):
        data = []
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
        return data

    def _show_domains(self, bounds=True):

        data = []
        nx, ny, nz = self.shape

        if not bounds:
            domains = [sub for sub in self.domains
                       if sub.ix[0] != 0 and sub.iy[0] != 0 and sub.iz[0] != 0
                       and sub.ix[1] != nx - 1 and sub.iy[1] != ny - 1 and sub.iz[1] != nz - 1]
        else:
            domains = self.domains

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

    def _show_mask(self, axis=0):
        data = []
        nx, ny, nz = self.shape
        x, y, z = _np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
        data.append(_go.Volume(x=x.flatten(),
                               y=y.flatten(),
                               z=z.flatten(),
                               value=self.mask[:, :, :, axis].flatten(),
                               isomin=-self.stencil,
                               isomax=self.stencil,
                               opacity=0.1,
                               opacityscale=[[-1, 1], [-0.5, 0], [0.5, 0], [1, 1]],
                               surface_count=16,
                               colorscale='RdBu',
                               flatshading=True,
                               caps=dict(x_show=False, y_show=False, z_show=False),
                               slices=dict(x_show=False, x_fill=0,
                                           y_show=False, y_fill=0,
                                           z_show=True, z_fill=1),
                               ))
        return data

    def zoom(self, ix=slice(None, None), iy=slice(None, None), iz=0, figsize=(15, 4)):
        """ Plot a zoom at (ix, ix, iz) of the mask. """

        _, axes = _plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(self.mask[ix, iy, iz, 0],
                       vmin=-self.stencil, vmax=self.stencil, origin="lower")
        axes[1].imshow(self.mask[ix, iy, iz, 1],
                       vmin=-self.stencil, vmax=self.stencil, origin="lower")
        axes[2].imshow(self.mask[ix, iy, iz, 2],
                       vmin=-self.stencil, vmax=self.stencil, origin="lower")

        _plt.show()
