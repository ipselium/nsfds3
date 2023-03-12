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
# Creation Date : 2022-06-09 - 22:15:26
"""
-----------

Some helper classes and functions to represent meshes graphically.

-----------
"""

import os
import sys
import pathlib
import numpy as _np

import matplotlib.pyplot as _plt
from matplotlib import patches as _patches
import matplotlib.animation as _ani
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.graph_objects as _go
from plotly.subplots import make_subplots

from progressbar import ProgressBar, Bar, ETA
from rich.progress import track

from mplutils import modified_jet, MidPointNorm, set_figsize, get_subplot_shape
from nsfds3.utils.data import DataExtractor, FieldExtractor, DataIterator, nearest_index
from libfds.fields import Fields2d, Fields3d


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


cmap = modified_jet()


class MPLViewer:
    """ """

    def __init__(self, cfg, msh, data):

        self.cfg = cfg
        self.msh = msh
        if isinstance(data, (Fields2d, Fields3d)):
            self.data = FieldExtractor(data)
        elif isinstance(data, (pathlib.Path, str)):
            self.data = DataExtractor(data)
        elif isinstance(data, DataExtractor):
            self.data = data
        else:
            raise ValueError('fld can be Fields2d, Fields3d, DataExtractor, or path to hdf5 file')

        self.volumic = True if len(cfg.shape) == 3 else False

    def show(self, view='p', vmin=None, vmax=None, show_nans=False, show_bz=False, slices=None, iteration=0, figsize=(9, 9)):

        var = self.data.get(view=view, iteration=iteration)

        if not vmin:
            vmin = _np.nanmin(var)
        if not vmax:
            vmax = _np.nanmax(var)

        norm = MidPointNorm(vmin=vmin, vmax=vmax, midpoint=0)

        fig, ax = self.frame(var, norm, show_nans=show_nans, show_bz=show_bz, slices=slices, figsize=figsize)

        _plt.show()

    def frame(self, var, norm, show_nans=False, show_bz=False, slices=None, iteration=0, figsize=(9, 9)):

        if len(self.msh.shape) == 3:
            return self._fields3d(var, norm, show_nans, show_bz, slices, figsize)
        else:
            return self._fields2d(var, norm, show_nans, show_bz, figsize)

    def _fields2d(self, var, norm, show_nans, show_bz, figsize=(9, 9)):
        """ Show 2d results. """

        # midpoint
        if norm.vmin > 0 and norm.vmax > 0:
            midpoint = _np.nanmean(var)
        else:
            midpoint = 0

        # ticks
        if abs(norm.vmin - midpoint) / norm.vmax > 0.33:
            ticks = [norm.vmin, midpoint, norm.vmax]
        else:
            ticks = [midpoint, norm.vmax]

        fig, ax = _plt.subplots(1, 1, figsize=figsize)

        im = ax.pcolorfast(self.msh.x, self.msh.y, var[:-1, :-1], cmap=cmap, norm=norm)
        ax.set_aspect(1.)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        _plt.colorbar(im, cax=cax, ticks=ticks)

        # Buffer zones
        bc = [self.msh.nbz if bc == 'A' else 0 for bc in self.msh.bc]
        origin, extrema = (bc[0], bc[2]), (bc[1], bc[3])
        size = [s - o - e for s, o, e in zip(self.msh.shape, origin, extrema)]
        size = [self.msh.axis[i][o + s - 1] - self.msh.axis[i][o]
                for i, (o, s) in enumerate(zip(origin, size))]
        origin = [self.msh.axis[i][j] for i, j in enumerate(origin)]
        edges = _patches.Rectangle(origin, *size, linewidth=3, fill=None)
        ax.add_patch(edges)
        if not show_bz:
            ax.set_xlim(origin[0], origin[0] + size[0])
            ax.set_ylim(origin[1], origin[1] + size[1])

        # Nan
        if show_nans:
            nans = _np.where(_np.isnan(var))[::-1]
            ax.plot(*nans, 'r.')

        # Obstacles
        for obs in self.msh.obstacles:
            origin = [self.msh.axis[i][j] for i, j in enumerate(obs.origin)]
            size = [self.msh.axis[i][o + s - 1] - self.msh.axis[i][o]
                    for i, (o, s) in enumerate(zip(obs.origin, obs.size))]
            edges = _patches.Rectangle(origin, *size, linewidth=3, fill=None)
            ax.add_patch(edges)

        return fig, im

    def _get_slices(self, slices):

        if slices:
            if all([a < b for a, b in zip(slices[::-1], self.msh.shape)]):
                self.i_xy, self.i_xz, self.i_zy = slices
            else:
                raise IndexError('Slices out of bounds')
        else:
            self.i_xy, self.i_xz, self.i_zy = [int(n/2) for n in self.msh.shape]

    def _fields3d(self, var, norm, show_nans, show_bz, slices, figsize=(9, 9)):
        """ Show 3d results. """

        foreground = {'linewidth': 3, 'fill': False}
        background = {'linewidth': 3, 'fill': True, 'facecolor': (0.8, 0.8, 1), 'alpha':0.2}

        self._get_slices(slices)

        # midpoint
        if norm.vmin > 0 and norm.vmax > 0:
            midpoint = _np.nanmean(var)
        else:
            midpoint = 0

        # ticks
        if abs(norm.vmin - midpoint) / norm.vmax > 0.33:
            ticks = [norm.vmin, midpoint, norm.vmax]
        else:
            ticks = [midpoint, norm.vmax]

        ims = []

        fig, ax_xy = _plt.subplots(figsize=figsize, tight_layout=True)

        # Sizes
        width, height = fig.get_size_inches()
        size_x = self.msh.x[-self.msh.nbz if self.msh.bc[1] == 'A' else -1] - self.msh.x[0 if self.msh.bc[0] == 'A' else 0]
        size_y = self.msh.y[-self.msh.nbz if self.msh.bc[3] == 'A' else -1] - self.msh.y[0 if self.msh.bc[2] == 'A' else 0]
        size_z = self.msh.z[-self.msh.nbz if self.msh.bc[5] == 'A' else -1] - self.msh.z[0 if self.msh.bc[4] == 'A' else 0]

        # xy plot:
        ims.append(ax_xy.pcolorfast(self.msh.x, self.msh.y, var[:-1, :-1, self.i_xy].T, cmap=cmap, norm=norm))
        ax_xy.plot(self.msh.x[[0, -1]], self.msh.y[[self.i_xz, self.i_xz]], color='gold', linewidth=1)
        ax_xy.plot(self.msh.x[[self.i_zy, self.i_zy]], self.msh.y[[0, -1]], color='green', linewidth=1)
        ax_xy.set_xlabel(r'$x$ [m]')
        ax_xy.set_ylabel(r'$y$ [m]')

        # position, size, pad
        divider = make_axes_locatable(ax_xy)
        ax_xz = divider.append_axes("top", height * size_z / (size_y + size_z), pad=0., sharex=ax_xy)
        ax_zy = divider.append_axes("right", 0.95 * width * size_z / (size_x + size_z), pad=0., sharey=ax_xy)

        # xz and zy plots
        ims.append(ax_xz.pcolorfast(self.msh.x, self.msh.z, var[:-1, self.i_xz, :-1].T, cmap=cmap, norm=norm))
        ax_xz.plot(self.msh.x[[0, -1]], self.msh.z[[self.i_xy, self.i_xy]], color='gold', linewidth=1)
        ax_xz.xaxis.set_tick_params(labelbottom=False)
        ax_xz.set_ylabel(r'$z$ [m]')

        ims.append(ax_zy.pcolorfast(self.msh.z, self.msh.y, var[self.i_zy, :-1, :-1], cmap=cmap, norm=norm))
        ax_zy.plot(self.msh.z[[self.i_xy, self.i_xy]], self.msh.y[[0, -1]], color='green', linewidth=1)
        ax_zy.yaxis.set_tick_params(labelleft=False)
        ax_zy.set_xlabel(r'$z$ [m]')

        #fig.subplots_adjust(hspace=0.01, wspace=0.01)

        for ax in fig.get_axes():
            ax.set_aspect(1.)

        ax_bar = divider.append_axes("right", size="5%", pad=0.)
        fig.colorbar(ims[0], cax=ax_bar, ticks=ticks)
        #ax_bar.xaxis.set_ticks_position("top")

        # Buffer zones
        bc = [self.msh.nbz if bc == 'A' else 0 for bc in self.msh.bc]
        origin, extrema = (bc[0], bc[2], bc[4]), (bc[1], bc[3], bc[5])
        size = [s - o - e for s, o, e in zip(self.msh.shape, origin, extrema)]
        size = [self.msh.axis[i][o + s - 1] - self.msh.axis[i][o]
                for i, (o, s) in enumerate(zip(origin, size))]
        origin = [self.msh.axis[i][j] for i, j in enumerate(origin)]

        edges = _patches.Rectangle((origin[0], origin[1]), size[0], size[1], **foreground)
        ax_xy.add_patch(edges)

        edges = _patches.Rectangle((origin[0], origin[2]), size[0], size[2], **foreground)
        ax_xz.add_patch(edges)

        edges = _patches.Rectangle((origin[2], origin[1]), size[2], size[1], **foreground)
        ax_zy.add_patch(edges)

        if not show_bz:
            ax_xy.set_xlim(origin[0], origin[0] + size[0])
            ax_xy.set_ylim(origin[1], origin[1] + size[1])

            ax_xz.set_xlim(origin[0], origin[0] + size[0])
            ax_xz.set_ylim(origin[2], origin[2] + size[2])

            ax_zy.set_xlim(origin[2], origin[2] + size[2])
            ax_zy.set_ylim(origin[1], origin[1] + size[1])

        # Obstacles
        for obs in self.msh.obstacles:

            origin = [self.msh.axis[i][j] for i, j in enumerate(obs.origin)]
            size = [self.msh.axis[i][o + s - 1] - self.msh.axis[i][o]
                    for i, (o, s) in enumerate(zip(obs.origin, obs.size))]

            kwargs = foreground if self.i_xy in obs.rz else background
            edges = _patches.Rectangle((origin[0], origin[1]), size[0], size[1], **kwargs)
            ax_xy.add_patch(edges)

            kwargs = foreground if self.i_xz in obs.ry else background
            edges = _patches.Rectangle((origin[0], origin[2]), size[0], size[2], **kwargs)
            ax_xz.add_patch(edges)

            kwargs = foreground if self.i_zy in obs.rx else background
            edges = _patches.Rectangle((origin[2], origin[1]), size[2], size[1], **kwargs)
            ax_zy.add_patch(edges)

        return fig, ims

    def _init_movie(self, view):

        title = os.path.basename(self.cfg.savefile).split('.')[0]
        views = {'p': r'$p_a$ [Pa]',
                 'e': r'$e$ [kg.m$^2$.s$^{-2}$]',
                 'vx': r'$v_x$ [m/s]',
                 'vy': r'$v_y$ [m/s]',
                 'vz': r'$v_y$ [m/s]',
                 'rho': r'$\rho$ [kg.m$^3$]',
                 're': r'$\rho e$ [kg$^2$.m$^{-1}$.s$^{-2}$]',
                 'ru': r'$\rho v_x$ [kg.m$^{-2}$/s]',
                 'rv': r'$\rho v_y$ [kg.m$^{-2}$/s]',
                 'rw': r'$\rho v_y$ [kg.m$^{-2}$/s]',
                 'vxyz': r'$\omega$ [m/s]'}

        metadata = dict(title=title, filename=f'{title}_{view}.mkv',
                        view=view, var=views[view], comment='Made with nsfds3')

        return metadata

    def movie(self, view='p', nt=None, ref=None, figsize=(9, 9),
              show_nans=False, show_bz=False, slices=None,
              dpi=100, fps=24):
        """ Make movie. """

        if not isinstance(self.data, DataExtractor):
            print('movie method only available for DataExtractor')
            sys.exit(1)

        # Nb of iterations and reference
        nt = self.cfg.nt if not nt else nearest_index(nt, self.cfg.ns, self.cfg.nt)
        ref = 'auto' if not ref else ref

        # Create Iterator and make 1st frame
        data = DataIterator(self.data, view=view, nt=nt)
        vmin, vmax = self.data.reference(view=view, ref=ref)
        norm = MidPointNorm(vmin=vmin, vmax=vmax, midpoint=0)
        i, var = next(data)
        fig, im = self.frame(var, norm,
                             show_nans=show_nans, show_bz=show_bz,
                             slices=slices, iteration=i, figsize=figsize)

        # Movie parameters
        metadata = self._init_movie(view)
        writer = _ani.FFMpegWriter(fps=fps, metadata=metadata, bitrate=-1, codec="libx264")
        with writer.saving(fig, self.cfg.savepath / metadata['filename'], dpi=dpi):

            writer.grab_frame()

            for i, var in track(data, description='Making movie...', disable=self.cfg.quiet):
                axes = fig.get_axes()
                if self.volumic:
                    im[0].set_data(self.msh.x, self.msh.y, var[:-1, :-1, self.i_xy].T)
                    im[1].set_data(self.msh.x, self.msh.z, var[:-1, self.i_xz, :-1].T)
                    im[2].set_data(self.msh.z, self.msh.y, var[self.i_zy, :-1, :-1])
                    axes[1].set_title(metadata['var'] + f' (n={i})')
                else:
                    im.set_data(self.msh.x, self.msh.y, var[:-1, :-1])
                    axes[0].set_title(metadata['var'] + f' (n={i})')

                writer.grab_frame()
            self.data.close()


class CDViewer:
    """ Computation domain viewer. """

    def __init__(self, obj):
        self.shape = obj.shape
        self.volumic = len(self.shape) == 3
        self.obstacles = obj.obstacles
        self.domains = obj.domains
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
            - BC profiles, figsize, Buffer Zone, probes, filename
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
            fig.add_traces(self._mesh3d(obstacles=obstacles, domains=domains, bounds=True))
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

            fig.add_traces(self._mesh3d(obstacles=obstacles, domains=domains, bounds=bounds),
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
