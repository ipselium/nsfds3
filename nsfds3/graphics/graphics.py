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
from scipy import signal as _signal

import matplotlib.pyplot as _plt
from matplotlib import patches as _patches, path as _path
import matplotlib.animation as _ani
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.graph_objects as _go
from plotly.subplots import make_subplots

from progressbar import ProgressBar, Bar, ETA
from rich.progress import track

from mplutils import modified_jet, MidPointNorm, set_figsize, get_subplot_shape
from nsfds3.utils.data import DataExtractor, FieldExtractor, DataIterator, nearest_index
from nsfds3.utils.misc import dict_update

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


class MeshViewer:
    """ """

    dkwargs = dict(figsize=(10, 10), dpi=100,
                   grid=True, buffer=True, obstacles=True, domains=False, N=1,
                   slices = None,
                   kwargs_grid=dict(),
                   kwargs_obstacles=dict(facecolor='k', alpha=0.2, fill=True, hatch='x'),
                   kwargs_domains=dict(facecolor='y'),
                   kwargs_buffer=dict(linewidth=3, edgecolor='k', fill=False))

    def __init__(self, msh):

        self.msh = msh
        if hasattr(self.msh, 'J'):
            self.grid = self.curvilinear_grid
            self.objects = self.curvilinear_objects
            self.axis = self.msh.paxis
        else:
            self.grid = self.cartesian_grid
            self.objects = self.cartesian_objects
            self.axis = self.msh.axis

    def show(self, **kwargs):

        kwargs = dict_update(self.dkwargs, kwargs)

        if self.msh.volumic:
            fig, *_ = self._frame3d(**kwargs)
        else:
            fig, *_ = self._frame2d(**kwargs)

        fig.show()

    def _frame3d(self, cbar=False, **kwargs):

        slices = self._get_slices(kwargs.get('slices'))

        fig, ax_xy = _plt.subplots(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'), tight_layout=True)

        # Size fig. & Limits
        width, height = fig.get_size_inches()
        limits = self.msh.domain_limits if kwargs.get('buffer') else self.msh.buffer_limits
        size_x = _np.diff(limits[0])[0]
        size_y = _np.diff(limits[1])[0]
        size_z = _np.diff(limits[2])[0]

        # Axes
        divider = make_axes_locatable(ax_xy)
        ax_xz = divider.append_axes("top", height * size_z / (size_y + size_z), pad=0., sharex=ax_xy)
        ax_zy = divider.append_axes("right", 0.95 * width * size_z / (size_x + size_z), pad=0., sharey=ax_xy)

        for ax in fig.get_axes():
            ax.set_aspect(1.)

        if cbar:
            ax_bar = divider.append_axes("right", size="5%", pad=0.1)
        else:
            ax_bar = None

        # Labels
        ax_xy.set_xlabel(r'$x$ [m]')
        ax_xy.set_ylabel(r'$y$ [m]')

        ax_xz.xaxis.set_tick_params(labelbottom=False)
        ax_xz.set_ylabel(r'$z$ [m]')

        ax_zy.yaxis.set_tick_params(labelleft=False)
        ax_zy.set_xlabel(r'$z$ [m]')

        # Cross section lines
        ax_xy.plot(self.msh.x[[0, -1]], self.msh.y[[self.i_xz, self.i_xz]], color='gold', linewidth=1)
        ax_xy.plot(self.msh.x[[self.i_zy, self.i_zy]], self.msh.y[[0, -1]], color='green', linewidth=1)
        ax_xz.plot(self.msh.x[[0, -1]], self.msh.z[[self.i_xy, self.i_xy]], color='gold', linewidth=1)
        ax_zy.plot(self.msh.z[[self.i_xy, self.i_xy]], self.msh.y[[0, -1]], color='green', linewidth=1)

        # Grid, Buffer zones, Obstacles, and domains
        obs_xy = ([obs for obs in self.msh.obstacles if self.i_xy in obs.rz],
                  [obs for obs in self.msh.obstacles if self.i_xy not in obs.rz])
        obs_xz = ([obs for obs in self.msh.obstacles if self.i_xz in obs.ry],
                  [obs for obs in self.msh.obstacles if self.i_xz not in obs.ry])
        obs_zy = ([obs for obs in self.msh.obstacles if self.i_zy in obs.rx],
                  [obs for obs in self.msh.obstacles if self.i_zy not in obs.rx])

        for ax, indices, obs in zip([ax_xy, ax_xz, ax_zy], [(0, 1), (0, 2), (2, 1)], [obs_xy, obs_xz, obs_zy]):

            if self.msh.mesh_type == 'Curvilinear':
                s = [slice(None) if i in indices else s for i, s in enumerate(slices)]
                args = ax, [v[tuple(s)] for v in self.axis], indices
            else:
                args = ax, self.axis, indices

            if kwargs.get('grid'):
                self.grid(*args, N=kwargs.get('N'), **kwargs.get('kwargs_grid'))

            if kwargs.get('obstacles'):
                self.objects(*args, obs[0], **kwargs.get('kwargs_obstacles'))
                self.objects(*args, obs[1], alpha=0.1, facecolor='k')

            if kwargs.get('buffer'):
                self.objects(*args, self.msh.buffer, **kwargs.get('kwargs_buffer'))
            else:
                ax.set_xlim(self.msh.buffer_limits[indices[0]])
                ax.set_ylim(self.msh.buffer_limits[indices[1]])

        return fig, ax_xy, ax_xz, ax_zy, ax_bar

    def _frame2d(self, cbar=False, **kwargs):

        fig, ax = _plt.subplots(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'), tight_layout=True)
        if cbar:
            divider = make_axes_locatable(ax)
            ax_bar = divider.append_axes("right", size="5%", pad=0.1)
        else:
            ax_bar = None

        args = ax, self.axis, (0, 1)

        if kwargs.get('grid'):
            self.grid(*args, N=kwargs.get('N'), **kwargs.get('kwargs_grid'))

        if kwargs.get('obstacles'):
            self.objects(*args, self.msh.obstacles, **kwargs.get('kwargs_obstacles'))

        if kwargs.get('domains'):
            self.objects(*args, self.msh.domains, **kwargs.get('kwargs_domains'))

        if kwargs.get('buffer'):
            self.objects(*args, self.msh.buffer, **kwargs.get('kwargs_buffer'))
        else:
            ax.set_xlim(self.msh.buffer_limits[0])
            ax.set_ylim(self.msh.buffer_limits[1])

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_aspect(1.)

        return fig, ax, ax_bar

    def _get_slices(self, slices):

        if slices:
            if all([a < b for a, b in zip(slices, self.msh.shape[::-1])]):
                self.i_xy, self.i_xz, self.i_zy = slices
            else:
                raise IndexError('Slices out of bounds')
        else:
            self.i_xy, self.i_xz, self.i_zy = [int(n/2) for n in self.msh.shape[::-1]]

        return self.i_xy, self.i_xz, self.i_zy

    @staticmethod
    def cartesian_objects(ax, axes, indices, obj, **kwargs):
        """ Plot cartesian objects on ax.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Axis where to draw the grid
        axes : tuple of np.ndarray
            Grid axes (x, y [, z])
        indices : tuple of int
            Indices of the axes to use
        obj : sequence
            Sequence of objects that must be drawn
        **kwargs : dict
            Keyword Arguments of matplotlib.patches.Rectangle
        """
        dkwargs = dict(linewidth=3, edgecolor='k', facecolor=None, alpha=1., fill=False, hatch=None)
        kwargs = dict_update(dkwargs, kwargs)

        if hasattr(obj, 'sid'):
            obj = (obj, )

        for obs in obj:
            sub_origin = [obs.origin[i] for i in indices]
            sub_size = [obs.size[i] for i in indices]
            origin = [axes[i][j] for i, j in zip(indices, sub_origin)]
            size = [axes[i][o + s - 1] - axes[i][o]
                    for i, o, s in zip(indices, sub_origin, sub_size)]
            edges = _patches.Rectangle(origin, *size, **kwargs)
            ax.add_patch(edges)

    @staticmethod
    def curvilinear_objects(ax, axes, indices, obj, **kwargs):
        """ Plot curvilinear objects on ax.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Axis where to draw the grid
        axes : tuple of np.ndarray
            Grid axes (x, y [, z])
        indices : tuple of int
            Indices of the axes to use
        obj : sequence
            Sequence of objects that must be drawn
        **kwargs : dict
            Keyword Arguments of matplotlib.patches.Rectangle
        """
        dkwargs = dict(linewidth=3, edgecolor='k', facecolor=None, alpha=1., fill=False, hatch=None)
        kwargs = dict_update(dkwargs, kwargs)

        if hasattr(obj, 'sid'):
            obj = (obj, )

        # Fix for when indices is (2, 1)
        if sorted(indices) == list(indices):
            u, v = axes[indices[0]], axes[indices[1]]
        else:
            u, v = axes[indices[0]].T, axes[indices[1]].T

        for sub in obj:
            b = [(u[i, sub.coords[indices[1]][0]], v[i, sub.coords[indices[1]][0]]) for i in sub.ranges[indices[0]]][::-1]
            l = [(u[sub.coords[indices[0]][0], i], v[sub.coords[indices[0]][0], i]) for i in sub.ranges[indices[1]]]
            t = [(u[i, sub.coords[indices[1]][1]], v[i, sub.coords[indices[1]][1]]) for i in sub.ranges[indices[0]]]
            r = [(u[sub.coords[indices[0]][1], i], v[sub.coords[indices[0]][1], i]) for i in sub.ranges[indices[1]]]

            verts = b + l + t + r
            codes = [_path.Path.MOVETO] + \
                    (len(verts)-2)*[_path.Path.LINETO] + \
                    [_path.Path.CLOSEPOLY]
            path = _path.Path(verts, codes)
            patch = _patches.PathPatch(path, **kwargs)
            ax.add_patch(patch)

    @staticmethod
    def cartesian_grid(ax, axes, indices, N, **kwargs):
        """ Plot cartesian grid on ax.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Axis where to draw the grid
        axes : tuple of np.ndarray
            Grid axes (x, y [, z])
        indices : tuple of int
            Indices of the axes to use
        N : int
            Plot a grid line every N points
        **kwargs : dict
            Keyword Arguments of ax.vlines/ax.hlines
        """

        dkwargs = dict(color='k', linewidth=0.1)
        kwargs = dict_update(dkwargs, kwargs)

        for i in _np.append(axes[indices[0]][::N], axes[indices[0]][-1]):
            ax.vlines(i, axes[indices[1]].min(), axes[indices[1]].max(), **kwargs)

        for i in _np.append(axes[indices[1]][::N], axes[indices[1]][-1]):
            ax.hlines(i, axes[indices[0]].min(), axes[indices[0]].max(), **kwargs)

    @staticmethod
    def curvilinear_grid(ax, axes, indices, N, **kwargs):
        """ Plot curvilinear grid on ax.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Axis where to draw the grid
        axes : tuple of np.ndarray
            Grid axes (x, y [, z])
        indices : tuple of int
            Indices of the axes to use
        N : int
            Plot a grid line every N points
        **kwargs : dict
            Keyword Arguments of ax.vlines/ax.hlines
        """

        dkwargs = dict(color='k', linewidth=0.1)
        kwargs = dict_update(dkwargs, kwargs)

        # Fix for when indices is (2, 1)
        if sorted(indices) == list(indices):
            u, v = axes[indices[0]], axes[indices[1]]
        else:
            u, v = axes[indices[0]].T, axes[indices[1]].T

        for i in _np.arange(0, v.shape[1], N):
            ax.plot(u[:, i], v[:, i], **kwargs)

        for i in _np.arange(0, u.shape[0], N):
            ax.plot(u[i, :], v[i, :], **kwargs)

        ax.plot(u[:, -1], v[:, -1], **kwargs)
        ax.plot(u[-1, :], v[-1, :], **kwargs)


class MPLViewer(MeshViewer):
    """
    TODO : Factoriser fields2d/fields3d avec MeshViewer._frame2d/frame3d

    """

    dkwargs = dict(figsize=(9, 9), dpi=100, fps=24,
                   grid=False, buffer=True, obstacles=True, domains=False,
                   probes=True, nans=False,
                   kwargs_grid=dict(),
                   kwargs_obstacles=dict(),
                   kwargs_domains=dict(facecolor='r', alpha=0.1), )

    def __init__(self, cfg, msh, data):

        super().__init__(msh)
        self.cfg = cfg

        if isinstance(data, (Fields2d, Fields3d)):
            self.data = FieldExtractor(data)
        elif isinstance(data, (pathlib.Path, str)):
            self.data = DataExtractor(data)
        elif isinstance(data, DataExtractor):
            self.data = data
        else:
            raise ValueError('fld can be Fields2d, Fields3d, DataExtractor, or path to hdf5 file')

        for name, ax in zip(('x', 'y', 'z'), msh.axis):
            setattr(self, name, ax)

    def show(self, view='p', vmin=None, vmax=None,  iteration=0, slices=None, **kwargs):


        kwargs = dict_update(self.dkwargs, kwargs)

        var = self.data.get(view=view, iteration=iteration)

        if not vmin:
            vmin = _np.nanmin(var)
        if not vmax:
            vmax = _np.nanmax(var)

        norm = MidPointNorm(vmin=vmin, vmax=vmax, midpoint=0)

        fig, ax = self.frame(var, norm, slices=slices, **kwargs)

        _plt.show()

    def frame(self, var, norm, slices=None, **kwargs):

        kwargs = dict_update(self.dkwargs, kwargs)

        if len(self.msh.shape) == 3:
            return self._fields3d(var, norm, slices, **kwargs)
        else:
            return self._fields2d(var, norm, **kwargs)

    def _fields2d(self, var, norm, **kwargs):
        """ Show 2d results. """

        fig, ax, ax_bar = self._frame2d(cbar=True, **kwargs)

        # Fill figure
        im = ax.pcolorfast(self.x, self.y, var[:-1, :-1], cmap=cmap, norm=norm)

        # Colorbar
        midpoint = _np.nanmean(var) if norm.vmin > 0 and norm.vmax > 0 else 0
        if abs(norm.vmin - midpoint) / norm.vmax > 0.33:
            ticks = [norm.vmin, midpoint, norm.vmax]
        else:
            ticks = [midpoint, norm.vmax]
        _plt.colorbar(im, cax=ax_bar, ticks=ticks)

        # Nans & Probes
        if kwargs.get('nans'):
            nans = _np.where(_np.isnan(var))[::-1]
            ax.plot(*nans, 'r.')

        if self.cfg.prb and kwargs.get('probes'):
            prbs = [(self.x[ix], self.y[iy]) for ix, iy in self.cfg.prb]
            for prb in prbs:
                ax.plot(*prb, 'ro')

        return fig, im

    def _fields3d(self, var, norm, slices, **kwargs):
        """ Show 3d results. """

        fig, ax_xy, ax_xz, ax_zy, ax_bar = self._frame3d(cbar=True, **kwargs)

        ims = []

        # Fill figure
        ims.append(ax_xy.pcolorfast(self.x, self.y, var[:-1, :-1, self.i_xy].T, cmap=cmap, norm=norm))
        ax_xy.plot(self.x[[0, -1]], self.y[[self.i_xz, self.i_xz]], color='gold', linewidth=1)
        ax_xy.plot(self.x[[self.i_zy, self.i_zy]], self.y[[0, -1]], color='green', linewidth=1)

        ims.append(ax_xz.pcolorfast(self.x, self.z, var[:-1, self.i_xz, :-1].T, cmap=cmap, norm=norm))
        ax_xz.plot(self.x[[0, -1]], self.z[[self.i_xy, self.i_xy]], color='gold', linewidth=1)

        ims.append(ax_zy.pcolorfast(self.z, self.y, var[self.i_zy, :-1, :-1], cmap=cmap, norm=norm))
        ax_zy.plot(self.z[[self.i_xy, self.i_xy]], self.y[[0, -1]], color='green', linewidth=1)

        # Colorbar
        midpoint = _np.nanmean(var) if norm.vmin > 0 and norm.vmax > 0 else 0
        if abs(norm.vmin - midpoint) / norm.vmax > 0.33:
            ticks = [norm.vmin, midpoint, norm.vmax]
        else:
            ticks = [midpoint, norm.vmax]
        fig.colorbar(ims[0], cax=ax_bar, ticks=ticks)

        # Probes
        if self.cfg.prb and kwargs.get('probes'):
            prbs = [(self.x[ix], self.y[iy], self.z[iz]) for ix, iy, iz in self.cfg.prb]
            for prb in prbs:
                color = 'r' if self.z[self.i_xy] == prb[2] else 'grey'
                ax_xy.plot(*(c for c in prb[:2]), marker='o', color=color)

                color = 'r' if self.y[self.i_xz] == prb[1] else 'grey'
                ax_xz.plot(*(c for i, c in enumerate(prb) if i != 1), marker='o', color=color)

                color = 'r' if self.x[self.i_zy] == prb[0] else 'grey'
                ax_zy.plot(*(c for c in prb[1:][::-1]), marker='o', color=color)

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
                 'wx': r'$\omega_x$ [m/s]',
                 'wy': r'$\omega_y$ [m/s]',
                 'wz': r'$\omega_z$ [m/s]'}

        metadata = dict(title=title, filename=f'{title}_{view}.mkv',
                        view=view, var=views[view], comment='Made with nsfds3')

        return metadata

    def movie(self, view='p', nt=None, ref=None, xlim=None, ylim=None, zlim=None,
              slices=None, **kwargs):
        """ Make movie. """

        kwargs = dict_update(self.dkwargs, kwargs)

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
        fig, im = self.frame(var, norm, slices=slices, iteration=i, **kwargs)

        # Movie parameters
        metadata = self._init_movie(view)
        writer = _ani.FFMpegWriter(fps=kwargs.get('fps'), metadata=metadata, bitrate=-1, codec="libx264")
        with writer.saving(fig, self.cfg.savepath / metadata['filename'], dpi=kwargs.get('dpi')):

            writer.grab_frame()

            for i, var in track(data, description='Making movie...', disable=self.cfg.quiet):
                axes = fig.get_axes()
                if self.msh.volumic:
                    im[0].set_data(self.x, self.y, var[:-1, :-1, self.i_xy].T)
                    im[1].set_data(self.x, self.z, var[:-1, self.i_xz, :-1].T)
                    im[2].set_data(self.z, self.y, var[self.i_zy, :-1, :-1])
                    axes[1].set_title(metadata['var'] + f' (n={i})')
                else:
                    im.set_data(self.x, self.y, var[:-1, :-1])
                    axes[0].set_title(metadata['var'] + f' (n={i})')

                writer.grab_frame()

    def probes(self, figsize=(9, 4)):
        """ Plot pressure at probes. """

        if not isinstance(self.data, DataExtractor):
            print('probes method only available for DataExtractor')
            sys.exit(1)

        probes = self.data.get_dataset('probe_locations').tolist()

        if not probes:
            return None

        p = self.data.get_dataset('probe_values')
        t = _np.arange(self.cfg.nt) * self.cfg.dt

        _, ax = _plt.subplots(figsize=figsize)
        for i, c in enumerate(probes):
            if self.cfg.mesh == 'curvilinear':
                p0 = self.cfg.p0 / self.msh.J[c[0], c[1]]
            else:
                p0 = self.data.get_attr('p0')
            ax.plot(t, p[i, :] - p0, label=f'@{tuple([self.axis[i][j] for i, j in enumerate(c)])}')
        ax.set_xlim(t.min(), t.max())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Pressure [Pa]')
        ax.legend()
        ax.grid()

        _plt.show()

        return None

    def spectrogram(self, M=None, figsize=(9, 4)):
        """ Plot spectograms at probes.

        Parameters
        ----------
        M : int
            Length of each segment
        """

        probes = self.data.get_dataset('probe_locations').tolist()

        if not probes:
            return None

        if not M:
            M = min(int(self.cfg.nt/20), 256)

        p = self.data.get_dataset('probe_values')

        fig, ax = _plt.subplots(p.shape[0], figsize=figsize)
        for i, c in enumerate(probes):

            if self.cfg.mesh == 'curvilinear':
                p0 = self.cfg.p0 / self.mesh.J[c[0], c[1]]
            else:
                p0 = self.cfg.p0

            freqs, times, Sx = _signal.spectrogram(p[i, :] - p0,
                                                   nperseg=M,
                                                   fs=1 / self.cfg.dt,
                                                   scaling='spectrum')
            Sx =  10 * _np.log10(Sx)
            im = ax[i].pcolormesh(times, freqs / 1000, Sx, cmap="Greys")
            ax[i].set_ylabel('Frequency [kHz]')
            if i != len(probes) - 1:
                ax[i].set_xticks([])

            fig.colorbar(im, ax=ax[i], label=f'probe {i}')

        ax[-1].set_xlabel('Time [s]')
        ax[0].set_title('Square spectrum magitude')
        _plt.tight_layout()

        _plt.show()

        return None


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
