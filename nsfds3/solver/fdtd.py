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
The `fdtd` module provides the `FDTD` (Finite Difference Time Domain) class that
setup and run the simulation
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

from nsfds3.graphics import MPLViewer
from nsfds3.utils import misc
from nsfds3.solver import CfgSetup



class FDTD:
    """ Solve Navier-Stokes equations using Finite Difference Time Domain (FDTD) technique.

    Parameters
    ----------
    cfg: CfgSetup
        Configuration of the simulation
    msh: CartesianGrid, CurvilinearGrid
        Grid used for the simulation
    quiet: bool
        If True, display informations on the standard output
    timings: bool
        If True, display complete timings during the simulation

    Notes
    -----
    When the simulation is complete, one can use the `show` method to display the desired field
    at the last iteration, or inspect the `fld` object that gathers all conservative variables
    at the last iteration.

    Finite differences schemes, Runge-Kutta algorithm and selective filter are applied using
    the technique described in [1]_. The shock capturing procedure is applied using the technique
    described in [2]_.

    References
    ----------

    .. [1] C. Bogey, C. Bailly, "A family of low dispersive and low dissipative explicit schemes for
           flow and noise computations", Journal of Computational Physics, Volume 194, Issue 1, 2004, 
           Pages 194-214.

    .. [2] C. Bogey, N. de Cacqueray, C. Bailly, "A shock-capturing methodology based on adaptative 
           spatial filtering for high-order non-linear computations", Journal of Computational Physics,
           Volume 228, Issue 5, 2009, Pages 1447-1465.
    """

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
        time = _np.linspace(0, cfg.nt * cfg.dt, cfg.nt + 1)
        for face in [f for f in msh.obstacles.faces if f.bc == "V"]:
            face.source_evolution = face.source_function(time)

        for source in self.cfg.src:
            source.set_evolution(time)

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

    def _log(self):
        """ Display informations about the simulation. """

        if self.timings:
            desc, time_per_iteration = misc.unload_timings(self._timings)

        if self.timings and not self.quiet:
            txt = f"Iteration: [red]{self.cfg.it:>6}\t[/]|\t"
            txt += f"Residuals: [green]{self.fld.residual():>.4f}\t[/]|\t"
            txt += f"Time: {time_per_iteration:>.4f}"
            print(Panel(txt))
            print(f"{desc}")

    def run(self):
        """ Run simulation. """
        ti = _pc()
        try:
            for self.cfg.it in track(range(self.cfg.it, self.cfg.nt + 1),
                                    disable=self.quiet):
                self._eulerian_fluxes()
                self._viscous_fluxes()
                self._selective_filter()
                self._shock_capture()
                self._vorticity()
                self._update_probes()
                if not self.cfg.it % self.cfg.ns:
                    self._save()
                    self._log()

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

    @misc.timer
    def _eulerian_fluxes(self):
        """ Compute Eulerian fluxes. """
        self.efluxes.rk4()

    @misc.timer
    def _viscous_fluxes(self):
        """ Compute viscous fluxes. """
        if self.cfg.vsc:
            self.vfluxes.integrate()
            self.efluxes.cout()

    @misc.timer
    def _selective_filter(self):
        """ Apply selective filter. """
        if self.cfg.flt:
            self.sfilter.apply()

    @misc.timer
    def _shock_capture(self):
        """ Apply shock capture procedure. """
        if self.cfg.cpt:
            self.scapture.apply()

    @misc.timer
    def _vorticity(self):
        """ Compute vorticity """
        if self.cfg.vrt:
            self.wxyz.compute()

    @misc.timer
    def _update_probes(self):
        """ Update probes. """
        if self.cfg.prb:
            for n, c in enumerate(self.cfg.prb):
                self.probes[n, self.cfg.it % self.cfg.ns] = self.fld.p[tuple(c)]

    @misc.timer
    def _save(self):
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
        """ Save cfg and msh objects. """

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
                _sys.exit(0)

        self.sfile = _h5py.File(self.cfg.datapath, 'w')
        self.sfile.attrs['vorticity'] = self.cfg.vrt
        self.sfile.attrs['ndim'] = self.msh.ndim
        self.sfile.attrs['p0'] = self.cfg.tp.p0
        self.sfile.attrs['gamma'] = self.cfg.tp.gamma

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
        self.sfile.attrs['rho0'] = self.cfg.tp.rho0
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

    from nsfds3.cpgrid import build_mesh

    config = CfgSetup()
    mesh = build_mesh(config)
    fdtd = FDTD(config, mesh)
    fdtd.run()
    fdtd.show()
