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
import h5py as _h5py

from libfds.fields import Fields
from libfds.fluxes import EulerianFluxes, ViscousFluxes, Vorticity
from libfds.filters import SelectiveFilter, ShockCapture


from rich import print
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import track
from rich.color import ANSI_COLOR_NAMES

from nsfds3.cpgrid import CartesianGrid
from nsfds3.solver import CfgSetup
from nsfds3.graphics import MPLViewer
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

        # Initialize sources (boundaries and domain)
        for face in [f for f in msh.obstacles.faces if f.bc == "V"]:
            face.source_evolution = face.source_function(cfg.nt, cfg.dt)

        for source in self.cfg.src:
            source.set_evolution(cfg.nt, cfg.dt)

        # Initialize solver
        self.fld = Fields(self.cfg, self.msh)
        self.efluxes = EulerianFluxes(self.fld)
        if self.cfg.vsc:
            self.vfluxes = ViscousFluxes(self.fld)
        if self.cfg.flt:
            self.sfilter = SelectiveFilter(self.fld)
        if self.cfg.cpt:
            self.scapture = ShockCapture(self.fld)
        if self.cfg.prb:
            self.probes = _np.zeros((len(cfg.prb), cfg.ns))
        if self.cfg.vrt:
            self.wxyz = Vorticity(self.fld)



        # Initialize save
        self._init_save()

        # Initialize timer
        self._timings = {}
        self.colors = [c for n, c in enumerate(ANSI_COLOR_NAMES) if 40 < n < 51]

    @staticmethod
    def timer(func):
        """ Time method of a class instance containing 'bench' attribute. """
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
        try:
            for self.cfg.it in track(range(self.cfg.it, self.cfg.nt + 1),
                                    disable=self.quiet):
                self.eulerian_fluxes()
                self.viscous_fluxes()
                self.selective_filter()
                self.shock_capture()
                self.vorticity()
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
        finally:
            self.sfile.close()
            self.save_objects()

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
    def vorticity(self):
        """ Compute vorticity """
        if self.cfg.vrt:
            self.wxyz.compute()

    @timer
    def update_probes(self):
        """ Update probes. """
        if self.cfg.prb:
            for n, c in enumerate(self.cfg.prb):
                self.probes[n, self.cfg.it % self.cfg.ns] = self.fld.p[tuple(c)]

    @timer
    def save(self):
        """ Save data. """

        self.sfile.attrs['itmax'] = self.cfg.it

        if self.cfg.save_fld:
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
            if self.msh.ndim == 3:
                self.sfile.create_dataset(f'rw_it{self.cfg.it}',
                                          data=self.fld.rw,
                                          compression=self.cfg.comp)

            if self.cfg.vrt:
                self.sfile.create_dataset(f'wz_it{self.cfg.it}',
                                        data=self.fld.wz,
                                        compression=self.cfg.comp)
                if self.msh.ndim == 3:
                    self.sfile.create_dataset(f'wx_it{self.cfg.it}',
                                            data=self.fld.wx,
                                            compression=self.cfg.comp)
                    self.sfile.create_dataset(f'wy_it{self.cfg.it}',
                                            data=self.fld.wy,
                                            compression=self.cfg.comp)
        if self.cfg.prb:
            self.sfile['probe_values'][:, self.cfg.it - self.cfg.ns:self.cfg.it] = self.probes

    def save_objects(self):

        with open(self.cfg.datapath.with_suffix('.cfg'), 'wb') as pkl:
            _pkl.dump(self.cfg, pkl, protocol=5)

        with open(self.cfg.datapath.with_suffix('.msh'), 'wb') as pkl:
            _pkl.dump(self.msh, pkl, protocol=5)

    def _init_save(self):
        """ Init save. """

        if self.cfg.datapath.is_file():
            msg = f'[bold red]{self.cfg.datapath}[/] already exists. \n[blink]Overwrite ?'
            overwrite = Prompt.ask(msg, choices=['yes', 'no'], default='no')
            if overwrite.lower() == 'no':
                _sys.exit(1)

        self.sfile = _h5py.File(self.cfg.datapath, 'w')
        self.sfile.attrs['vorticity'] = self.cfg.vrt
        self.sfile.attrs['ndim'] = self.msh.ndim
        self.sfile.attrs['p0'] = self.cfg.p0
        self.sfile.attrs['gamma'] = self.cfg.gamma

        # Not necessary ?
        self.sfile.attrs['obstacles'] = self.msh.get_obstacles()
        self.sfile.create_dataset('x', data=self.msh.x, compression=self.cfg.comp)
        self.sfile.create_dataset('y', data=self.msh.y, compression=self.cfg.comp)
        self.sfile.attrs['dx'] = self.msh.dx
        self.sfile.attrs['dy'] = self.msh.dy
        self.sfile.attrs['dt'] = self.cfg.dt
        self.sfile.attrs['nx'] = self.msh.nx
        self.sfile.attrs['ny'] = self.msh.ny
        self.sfile.attrs['nt'] = self.cfg.nt
        self.sfile.attrs['ns'] = self.cfg.ns
        self.sfile.attrs['rho0'] = self.cfg.rho0
        self.sfile.attrs['bz_n'] = self.cfg.bz_n
        self.sfile.attrs['mesh'] = self.msh.mesh_type
        self.sfile.attrs['bc'] = self.cfg.bc
        self.sfile.attrs['itmax'] = self.cfg.it
        if self.msh.ndim == 3:
            self.sfile.attrs['dz'] = self.msh.dz
            self.sfile.attrs['nz'] = self.msh.nz
            self.sfile.create_dataset('z', data=self.msh.z, compression=self.cfg.comp)

        probes = _np.zeros((len(self.cfg.prb), self.cfg.nt))
        self.sfile.create_dataset('probe_locations', data=self.cfg.prb)
        self.sfile.create_dataset('probe_values', data=probes,
                                  compression=self.cfg.comp)

        if self.msh.mesh_type.lower() == 'curvilinear':
            self.sfile.create_dataset('J', data=self.msh.J, compression=self.cfg.comp)
            #self.sfile.create_dataset('xn', data=self.msh.xn, compression=self.cfg.comp)
            #self.sfile.create_dataset('yn', data=self.msh.yn, compression=self.cfg.comp)
            self.sfile.create_dataset('xp', data=self.msh.xp, compression=self.cfg.comp)
            self.sfile.create_dataset('yp', data=self.msh.yp, compression=self.cfg.comp)
            if self.msh.ndim == 3:
                #self.sfile.create_dataset('zn', data=self.msh.yn, compression=self.cfg.comp)
                self.sfile.create_dataset('zp', data=self.msh.yp, compression=self.cfg.comp)

    def show(self, view='p', vmin=None, vmax=None, **kwargs):
        """ Show results. """
        viewer = MPLViewer(self.cfg, self.msh, self.fld)
        viewer.show(view=view, vmin=vmin, vmax=vmax, **kwargs)


if __name__ == '__main__':
    config = CfgSetup()
    args, kwargs = config.get_config()
    mesh = CartesianGrid(*args, **kwargs)
    fdtd = FDTD(config, mesh)
    print(mesh)
    print(mesh.domains)
    fdtd.run()
    fdtd.show()
