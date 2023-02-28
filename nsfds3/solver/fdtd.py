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
# Creation Date : 2022-07-11 - 22:25:34
# pylint: disable=redefined-builtin
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
"""
-----------
DOCSTRING

-----------
"""

import sys as _sys
import itertools as _it
import pickle as _pkl
from time import perf_counter as _pc
import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib import patches as _patches, path as _path
import h5py as _h5py

from libfds.fields import Fields
from libfds.fluxes import EulerianFluxes, ViscousFluxes
from libfds.filters import SelectiveFilter, ShockCapture
from mplutils.custom_cmap import modified_jet, MidPointNorm

from rich import print
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import track
from rich.color import ANSI_COLOR_NAMES

from nsfds3.cpgrid import RegularMesh
from nsfds3.solver import CfgSetup
from nsfds3.utils import misc


class FDTD:
    """ FDTD technique. """

    def __init__(self, cfg, msh, quiet=None, timings=None):

        # Initialize configuration & mesh
        self.cfg = cfg
        self.msh = msh

        # Arguments
        if isinstance(quiet, bool):
            self.quiet = quiet
        else:
            self.quiet = self.cfg.quiet

        if isinstance(timings, bool):
            self.timings = timings
        else:
            self.timings = self.cfg.timings

        # Initialize solver
        self.fld = Fields(self.cfg, self.msh)
        self.efluxes = EulerianFluxes(self.fld)
        if self.cfg.vsc:
            self.vfluxes = ViscousFluxes(self.fld) 
        if self.cfg.flt:
            self.sfilter = SelectiveFilter(self.fld)
        if self.cfg.cpt:
            self.scapture = ShockCapture(self.fld)
        if self.cfg.probes:
            self.probes = _np.zeros((len(cfg.probes), cfg.ns))
        if self.cfg.save_vortis:
            pass

        # Curvilinear stuff
        self._physical = False

        # Initialize save
        self._init_save()

        # Initialize timer
        self._timings = {}
        self.colors = [c for n, c in enumerate(ANSI_COLOR_NAMES) if 40 < n < 51]

        [f'yellow{i}' for i in range(1, 5)] + \
                      ['green1', 'green3', 'green4']

    @staticmethod
    def timer(func):
        """ Time method of a class containing 'bench' attribute. """
        def wrapper(self, *args, **kwargs):
            if self.timings:
                start = _pc()
            func(self, *args, **kwargs)
            if self.timings:
                if func.__name__ in self._timings:
                    self._timings[func.__name__].append(_pc() - start)
                else:
                    self._timings[func.__name__] = [_pc() - start, ]
        return wrapper

    def unload_timings(self):
        """ Empty timings and display. """

        desc = ""
        ttime = sum(list(_it.chain(*self._timings.values()))) / self.cfg.ns
        for color, key in zip(self.colors, self._timings):
            time = _np.array(self._timings[key]).mean()
            if time > 2e-4:
                desc += f'\t-[italic {color}]{key:20}: '
                desc += f'{time:.4f}\n'
            self._timings[key] = []

        if self.timings and not self.quiet:
            txt = f"Iteration: [red]{self.cfg.it:>6}\t[/]|\t"
            txt += f"Residuals: [green]{self.fld.residual():>.4f}\t[/]|\t"
            txt += f"Time: {ttime:>.4f}"
            print(Panel(txt))
            print(f"{desc}")

    @timer
    def run(self):
        """ Run FDTD. """
        ti = _pc()
        for self.cfg.it in track(range(self.cfg.it, self.cfg.nt + 1),
                                 disable=self.quiet):
            self.eulerian_fluxes()
            self.viscous_fluxes()
            self.toggle_system()
            self.selective_filter()
            self.shock_capture()
            self.toggle_system()
            self.update_vorticity()
            self.update_probes()
            if not self.cfg.it % self.cfg.ns:
                self.save()
                self.unload_timings()

        if not self.quiet:
            msg = 'Simulation completed in [red]{}[/].\n'
            msg += 'Final residuals of [red]{:>.4f}[/].\n'
            msg += 'End at physical time [red]t = {:.4f} sec.'
            print(Panel(msg.format(misc.secs_to_dhms(_pc() - ti),
                                   self.fld.residual(),
                                   self.cfg.dt * self.cfg.it)))

        self.sfile.close()

    @timer
    def eulerian_fluxes(self):
        """ Compute Eulerian fluxes. """
        self.efluxes.rk4()

    @timer
    def viscous_fluxes(self):
        """ Compute viscous fluxes. """
        if self.cfg.vsc:
            self.vfluxes.integrate()
            self.efluxes.cout()

    @timer
    def selective_filter(self):
        """ Apply selective filter. """
        if self.cfg.flt:
            self.sfilter.apply()

    @timer
    def shock_capture(self):
        """ Apply shock capture procedure. """
        if self.cfg.cpt:
            self.scapture.apply()

    @timer
    def toggle_system(self):
        """ Convert curvilinear coordinates : from physical to numeric or reverse. """
        if self.cfg.mesh == 'curvilinear':
            if self._physical:
                self.fld.phys2num()
                self._physical = not self._physical
            else:
                self.fld.num2phys()
                self._physical = not self._physical

    @timer
    def update_vorticity(self):
        """ Compute vorticity """
        if self.cfg.save_vortis:
            pass

    @timer
    def update_probes(self):
        """ Update probes. """
        if self.cfg.probes:
            for n, c in enumerate(self.cfg.probes):
                self.probes[n, self.cfg.it % self.cfg.ns] = self.fld.p[tuple(c)]

    @timer
    def save(self):
        """ Save data. """

        self.sfile.attrs['itmax'] = self.cfg.it

        if self.cfg.save_fields:
            self.sfile.create_dataset(f'r_it{self.cfg.it}',
                                      data=self.fld.r,
                                      compression=self.cfg.comp)
            self.sfile.create_dataset(f'ru_it{self.cfg.it}',
                                      data=self.fld.ru,
                                      compression=self.cfg.comp)
            self.sfile.create_dataset(f'rv_it{self.cfg.it}',
                                      data=self.fld.rv,
                                      compression=self.cfg.comp)
            self.sfile.create_dataset(f're_it{self.cfg.it}',
                                      data=self.fld.re,
                                      compression=self.cfg.comp)
            if self.msh.volumic:
                self.sfile.create_dataset(f'rw_it{self.cfg.it}',
                                          data=self.fld.rw,
                                          compression=self.cfg.comp)

        if self.cfg.save_vortis:
            self.sfile.create_dataset(f'w_it{self.cfg.it}',
                                      data=self.fld.w,
                                      compression=self.cfg.comp)

        if self.cfg.probes:
            self.sfile['probe_values'][:, self.cfg.it - self.cfg.ns:self.cfg.it] = self.probes

    def _init_save(self):
        """ Init save. """

        if self.cfg.datafile.is_file():
            msg = f'[bold red]{self.cfg.datafile}[/] already exists. \n[blink]Overwrite ?'
            overwrite = Prompt.ask(msg, choices=['yes', 'no'], default='no')
            if overwrite.lower() == 'no':
                _sys.exit(1)

        with open(self.cfg.datafile.with_suffix('.cfg'), 'wb') as pkl:
            _pkl.dump(self.cfg, pkl)

        self.sfile = _h5py.File(self.cfg.datafile, 'w')
        self.sfile.attrs['obstacles'] = self.msh.get_obstacles()
#        self.sfile.attrs['domains'] = self.msh.get_domains(only_xz=True)
        self.sfile.create_dataset('x', data=self.msh.x, compression=self.cfg.comp)
        self.sfile.create_dataset('y', data=self.msh.y, compression=self.cfg.comp)
        self.sfile.attrs['dx'] = self.msh.dx
        self.sfile.attrs['dy'] = self.msh.dy
        self.sfile.attrs['dt'] = self.cfg.dt
        self.sfile.attrs['nx'] = self.msh.nx
        self.sfile.attrs['ny'] = self.msh.ny
        self.sfile.attrs['nt'] = self.cfg.nt
        self.sfile.attrs['ns'] = self.cfg.ns
        self.sfile.attrs['p0'] = self.cfg.p0
        self.sfile.attrs['rho0'] = self.cfg.rho0
        self.sfile.attrs['gamma'] = self.cfg.gamma
        self.sfile.attrs['Npml'] = self.cfg.Npml
        self.sfile.attrs['mesh'] = self.cfg.mesh
        self.sfile.attrs['bc'] = self.cfg.bc
        self.sfile.attrs['itmax'] = self.cfg.it
        if self.msh.volumic:
            self.sfile.attrs['dz'] = self.msh.dz
            self.sfile.attrs['nz'] = self.msh.nz
            self.sfile.create_dataset('z', data=self.msh.z, compression=self.cfg.comp)

        probes = _np.zeros((len(self.cfg.probes), self.cfg.nt))
        self.sfile.create_dataset('probe_locations', data=self.cfg.probes)
        self.sfile.create_dataset('probe_values', data=probes,
                                  compression=self.cfg.comp)

        if self.cfg.mesh == 'curvilinear':
            self.sfile.create_dataset('J', data=self.msh.J, compression=self.cfg.comp)
            self.sfile.create_dataset('xn', data=self.msh.xn, compression=self.cfg.comp)
            self.sfile.create_dataset('yn', data=self.msh.yn, compression=self.cfg.comp)
            self.sfile.create_dataset('xp', data=self.msh.xp, compression=self.cfg.comp)
            self.sfile.create_dataset('yp', data=self.msh.yp, compression=self.cfg.comp)
            if self.msh.volumic:
                self.sfile.create_dataset('zn', data=self.msh.yn, compression=self.cfg.comp)
                self.sfile.create_dataset('zp', data=self.msh.yp, compression=self.cfg.comp)

    def show(self, variable='p', vmin=None, vmax=None, show_nans=False, slices=None):
        """ Show results. """

        if variable in ['p', 'ru', 'rv', 're', 'r']:
            var = _np.array(getattr(self.fld, variable))
        else:
            raise Exception('var must be p, ru, rv, re, or r')

        if variable == 'p':
            var -= self.cfg.p0

        if not vmin:
            vmin = _np.nanmin(var)
        if not vmax:
            vmax = _np.nanmax(var)

        cmap = modified_jet()
        norm = MidPointNorm(vmin=vmin, vmax=vmax, midpoint=0)

        if self.cfg.volumic:
            self._show3d(var, cmap, norm, show_nans=show_nans, slices=slices)
        else:
            self._show2d(var, cmap, norm, show_nans=show_nans)

    def _show2d(self, var, cmap, norm, show_nans=False):
        """ Show 2d results. """
        _, axes = _plt.subplots(1, 1, figsize=(9, 4))

        axes.imshow(var, origin='lower', cmap=cmap, norm=norm)

        if show_nans:
            nans = _np.where(_np.isnan(var))[::-1]
            axes.plot(*nans, 'r.')

        for obs in self.msh.obstacles:
            edges = _patches.Rectangle(obs.origin[::-1],
                                       *(_np.array(obs.size[::-1]) - 1),
                                       linewidth=3, fill=None)
            axes.add_patch(edges)
        _plt.show()

    def _show3d(self, var, cmap, norm, show_nans=False, slices=None):
        """ Show 3d results. """
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if slices:
            if all([a < b for a, b in zip(slices[::-1], self.msh.shape)]):
                i_xy, i_xz, i_zy = slices
            else:
                raise IndexError('Slices out of bounds')
        else:
            i_xy, i_xz, i_zy = self.cfg.izS, self.cfg.iyS, self.cfg.ixS

        fig, ax_xy = _plt.subplots(figsize=(9, 9), tight_layout=True)

        # Sizes
        width, height = fig.get_size_inches()
        size_x = self.msh.x[-1] - self.msh.x[0]
        size_y = self.msh.y[-1] - self.msh.y[0]
        size_z = self.msh.z[-1] - self.msh.z[0]

        # xy plot:
        ax_xy.pcolorfast(self.msh.x, self.msh.y, var[:, :, i_xy], cmap=cmap, norm=norm)
        ax_xy.plot(self.msh.x[[0, -1]], self.msh.y[[i_xz, i_xz]], color='gold', linewidth=1)
        ax_xy.plot(self.msh.x[[i_zy, i_zy]], self.msh.y[[0, -1]], color='green', linewidth=1)

        # create new axes on the right and on the top of the current axes
        divider = make_axes_locatable(ax_xy)
        # below height and pad are in inches
        ax_xz = divider.append_axes("top", 1.25*width*(size_x/size_z - 1), pad=0., sharex=ax_xy)    # position, size, pad
        ax_zy = divider.append_axes("right", 1.25*width*(size_x/size_z - 1), pad=0., sharey=ax_xy)

        # xz and zy plots
        ax_xz.pcolorfast(self.msh.x, self.msh.z, var[:, i_xz, :].T, cmap=cmap, norm=norm)
        ax_xz.plot(self.msh.x[[0, -1]], self.msh.z[[i_xy, i_xy]], color='gold', linewidth=1)
        ax_xz.xaxis.set_tick_params(labelbottom=False)

        ax_zy.pcolorfast(self.msh.z, self.msh.y, var[i_zy, :, :], cmap=cmap, norm=norm)
        ax_zy.plot(self.msh.z[[i_xy, i_xy]], self.msh.y[[0, -1]], color='green', linewidth=1)
        ax_zy.yaxis.set_tick_params(labelleft=False)

        for ax in fig.get_axes():
            ax.set_aspect(1.)

        # Obstacles
        facecolor = (0.8, 0.8, 1)
        alpha = 0.2
    
        for obs in self.msh.obstacles:
            if i_xy in obs.rz:
                edges = _patches.Rectangle((self.msh.x[obs.origin[0]], self.msh.y[obs.origin[1]]),
                                        obs.size[0] - 1, obs.size[1] - 1,
                                        linewidth=3, fill=False)
            else:
                edges = _patches.Rectangle((self.msh.x[obs.origin[0]], self.msh.y[obs.origin[1]]),
                                        obs.size[0] - 1, obs.size[1] - 1,
                                        linewidth=1, fill=True, facecolor=facecolor, alpha=alpha)
            ax_xy.add_patch(edges)
            
            if i_xz in obs.ry:
                edges = _patches.Rectangle((self.msh.x[obs.origin[0]], self.msh.y[obs.origin[2]]),
                                        obs.size[0] - 1, obs.size[2] - 1,
                                        linewidth=3, fill=False)
            else:
                edges = _patches.Rectangle((self.msh.x[obs.origin[0]], self.msh.y[obs.origin[2]]),
                                        obs.size[0] - 1, obs.size[2] - 1,
                                        linewidth=1, fill=True, facecolor=facecolor, alpha=alpha)
            ax_xz.add_patch(edges)
            
            if i_zy in obs.rx:
                edges = _patches.Rectangle((self.msh.x[obs.origin[2]], self.msh.y[obs.origin[1]]),
                                        obs.size[2] - 1, obs.size[1] - 1,
                                        linewidth=3, fill=False)
            else:
                edges = _patches.Rectangle((self.msh.x[obs.origin[2]], self.msh.y[obs.origin[1]]),
                                        obs.size[2] - 1, obs.size[1] - 1,
                                        linewidth=1, fill=True, facecolor=facecolor, alpha=alpha)
            ax_zy.add_patch(edges)

        _plt.show()


if __name__ == '__main__':
    config = CfgSetup()
    args, kwargs = config.get_config()
    mesh = RegularMesh(*args, **kwargs)
    fdtd = FDTD(config, mesh)
    print(mesh)
    print(mesh.domains)
    fdtd.run()
    fdtd.show()