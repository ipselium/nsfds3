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

from libfds.fields import Fields2d
from libfds.fluxes import EulerianFluxes2d
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
        self.fld = Fields2d(self.cfg, self.msh)
        self.efluxes = EulerianFluxes2d(self.fld)
        if self.cfg.vsc:
            pass
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
            msg += 'End at physical time [red]t = {:.4f} sec.'
            print(Panel(msg.format(misc.secs_to_dhms(_pc() - ti),
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
            pass

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
            self.sfile.create_dataset('r_it' + str(self.cfg.it),
                                      data=self.fld.r,
                                      compression=self.cfg.comp)
            self.sfile.create_dataset('ru_it' + str(self.cfg.it),
                                      data=self.fld.ru,
                                      compression=self.cfg.comp)
            self.sfile.create_dataset('rv_it' + str(self.cfg.it),
                                      data=self.fld.rv,
                                      compression=self.cfg.comp)
            self.sfile.create_dataset('re_it' + str(self.cfg.it),
                                      data=self.fld.re,
                                      compression=self.cfg.comp)
            if self.msh.volumic:
                self.sfile.create_dataset('rw_it' + str(self.cfg.it),
                                          data=self.fld.re,
                                          compression=self.cfg.comp)

        if self.cfg.save_vortis:
            self.sfile.create_dataset('w_it' + str(self.cfg.it),
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
#        self.sfile.attrs['obstacles'] = self.msh.get_obstacles()
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

    def show(self, variable='p', vmin=None, vmax=None, nans=False):
        """ Show results. """
        _, axes = _plt.subplots(1, 1, figsize=(9, 4))

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

        axes.imshow(var, origin='lower', cmap=cmap, norm=norm)

        if nans:
            nans = _np.where(_np.isnan(var))[::-1]
            axes.plot(*nans, 'r.')

        for obs in self.msh.obstacles:
            edges = _patches.Rectangle(obs.origin[::-1],
                                       *(_np.array(obs.size[::-1]) - 1),
                                       linewidth=3, fill=None)
            axes.add_patch(edges)
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
