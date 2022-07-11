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


import itertools as _it
from time import perf_counter as _pc
import numpy as np
import matplotlib.pyplot as plt

from libfds.fields import Fields2d
from libfds.fluxes import EulerianFluxes2d
from libfds.filters import SelectiveFilter, ShockCapture
from mplutils.custom_cmap import modified_jet, MidPointNorm

from rich import print
from rich.panel import Panel
from rich.progress import track


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
            self.probes = np.zeros((len(cfg.probes), cfg.ns))
        if self.cfg.save_vortis:
            pass

        # Initialize timer
        self._timings = {}
        self.colors = [f'yellow{i}' for i in range(1, 5)]

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
            desc += f'\t-[italic {color}]{key:20}: '
            desc += f'{np.array(self._timings[key]).mean():.4f}\n'
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
            self.selective_filter()
            self.shock_capture()
            if not self.cfg.it % self.cfg.ns:
                self.unload_timings()

        if not self.quiet:
            msg = 'Simulation completed in [red]{}[/].\n'
            msg += 'End at physical time [red]t = {:.4f} sec.'
            print(Panel(msg.format(misc.secs_to_dhms(_pc() - ti),
                                   self.cfg.dt * self.cfg.it)))

    @timer
    def eulerian_fluxes(self):
        """ Compute Eulerian fluxes. """
        self.efluxes.rk4()

    @timer
    def viscous_fluxes(self):
        """ Compute viscous fluxes """
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
    def update_pressure(self):
        """ Compute viscous fluxes """

    @timer
    def update_vorticity(self):
        """ Compute viscous fluxes """
        if self.cfg.save_vortis:
            pass

    @timer
    def update_probes(self):
        """ Update probes. """
        if self.cfg.probes:
            for n, c in enumerate(self.cfg.probes):
                self.probes[n, self.cfg.it % self.cfg.ns] = \
                        self.fld.p[c[0], c[1]]

    @timer
    def save(self):
        """ Compute viscous fluxes """

    def show(self):
        """ Show results. """
        _, axes = plt.subplots(1, 1, figsize=(9, 4))
        p = np.array(self.fld.p) - self.cfg.p0

        cmap = modified_jet()
        norm = MidPointNorm(vmin=p.min(), vmax=p.max(), midpoint=0)

        axes.imshow(p, origin='lower', cmap=cmap, norm=norm)
        plt.show()


if __name__ == '__main__':
    config = CfgSetup()
    args, kwargs = config.get_config()
    mesh = RegularMesh(*args, **kwargs)
    fdtd = FDTD(config, mesh)
    print(mesh)
    print(mesh.domains)
    fdtd.run()
    fdtd.show()
