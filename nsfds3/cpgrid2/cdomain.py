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
# Creation Date : 2022-06-09 - 23:00:01
"""
-----------

Module `cdomain` provides `ComputationDomains` class aiming at dividing the
grid into subdomains corresponding to different geometric configurations.

-----------
"""

import itertools as _it
import numpy as _np
import rich.progress as _rp
from .geometry import ObstacleSet, DomainSet, Domain, Box
from nsfds3.utils.misc import locations_to_cuboids, unique, Schemes, scheme_to_str, buffer_kwargs
import nsfds3.graphics as _graphics
from libfds.cutils import nonzeros, where
import time

class ComputationDomains:
    """ Divide computation domain in several Subdomains based on obstacles in presence.

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with 3 int objects.
    obstacles : list, :py:class:`nsfds3.mesher.geometry.ObstacleSet`, optional
        Obstacles in the computation domain.
    bc : {'[APW][APW][APW][APW]'}, optional
        Boundary conditions. Must be a 4 or 6 characters string corresponding to
        left, right, front, back, bottom, and top boundaries, respectively.
    nbz : int, optional
        Size of the buffer zone
    stencil : int, optional
        Size of the finite difference stencil.
    free: bool, optional
        Free memory after the domains are found

    TODO
    ----
        - optimizations :
            - locations_to_cuboids
    """

    _BC_U = ['W', 'A']
    _BC_C = ['P', ]

    def __init__(self, shape, obstacles=None, bc='WWWWWW', nbz=20, stencil=11, free=True):

        self.shape = shape
        self.ndim = len(shape)
        self.bc = bc
        self.nbz = nbz
        self.stencil, self._midstencil = stencil, int((stencil - 1) / 2)

        if isinstance(obstacles, ObstacleSet):
            self.obstacles = obstacles
        else:
            self.obstacles = ObstacleSet(shape, bc=bc, subs=obstacles, stencil=stencil)

        self.bounds = self.obstacles.bounds
        self.buffer = Domain(**buffer_kwargs(self.bc, self.nbz, self.shape))
        self._domains = [[] for i in range(self.ndim)]

        self.find_domains()

        if free:
            self._free()

    def _mask_init(self):
        """
        Initialize a mask that will contain 0 at the location of obstacles and 1 elsewhere.
        """
        self._mask = _np.ones(self.shape + (self.ndim, ), dtype=_np.int8)

        for obs in self.obstacles:
            self._mask[obs.sin + (slice(0, self.ndim), )] = 0

        # Fix covered faces
        for f in self.obstacles.covered:
            self._mask[f.sin + (slice(0, self.ndim), )] = 0

        # Fix junction between face to face overlapped objects
        combs = [(f1, f2) for f1, f2 in _it.combinations(self.obstacles.faces, r=2) if f1.sid != f2.sid]
        for f1, f2 in combs:
            if f1.intersects(f2) and f1.side == f2.opposite:
                s = tuple(zip(*f1.inner_indices().intersection(f2.inner_indices())))
                self._mask[s + (slice(0, self.ndim), )] = 0

        # Fix periodic faces
        for f in self.obstacles.periodic:
            self._mask[f.sin + (slice(0, self.ndim), )] = 0
            if f.colinear:
                fix = set()
                for fc in f.colinear:
                    fix |= set(f.intersection(fc))
                self._mask[tuple(zip(*fix)) + (slice(0, self.ndim), )] = 0

    def _mask_setup(self):
        """
        Fill a mask according to finite difference scheme to be applied on each point.
            - Obstacles        : 0
            - Centered schemes : 1
            - Forward scheme   : 11
            - Backward scheme  : -11
        """

        self._mask_init()

        for f in self.obstacles.uncentered:
            fbox = f.box(self._midstencil)
            base = _np.zeros(fbox.size, dtype=_np.int8)
            base[(slice(None), ) * self.ndim] = self._mask[f.base_slice + (f.axis, )] == 0
            base[self._mask[fbox.sn + (f.axis, )] == 1] *= f.normal * self.stencil
            base[base == 0] = 1
            self._mask[fbox.sn + (f.axis, )] *= base
            #print(_np.array((self._mask[f.sn + (f.axis, )] == f.normal * self.stencil).nonzero(), dtype=_np.int16).T)
            #self._domains[f.axis].append(Domain.from_slices(fbox.sn, env=self.shape))

        for f in [b for b in self.bounds if b.bc in self._BC_U]:
            sn = f.box(self._midstencil).sn
            self._mask[sn + (f.axis, )] = f.normal * self.stencil

            for o in f.overlapped:
                sn = o.inner_slices(f.axis)
                self._mask[sn + (f.axis, )] = 0

    def find_domains(self):

        self._mask_setup()
        confs = [1, -11, 11]

        with _rp.Progress(_rp.TextColumn("[bold blue]{task.description:<20}...", justify="right"),
                          _rp.BarColumn(bar_width=None),
                          _rp.TextColumn("[progress.percentage]{task.percentage:>3.1f}% •"),
                          _rp.TimeRemainingColumn(),
                          _rp.TextColumn("• {task.fields[details]}")) as pbar:

            task = pbar.add_task("[red]Building domains...",
                                 total=len(confs) * self.ndim, details='Starting...')

            for ax, n in zip('xyz', range(self.ndim)):
                for name, c in zip('c+-', confs):
                    ti = pbar.get_time()

                    # To optimize ?
                    idx = _np.array((self._mask[..., n] == c).nonzero(), dtype=_np.int16).T
                    cuboids = locations_to_cuboids(_np.ascontiguousarray(idx))
                    for cub in cuboids:
                        self._domains[n].append(Domain(cub['origin'], cub['size'], self.shape, tag=c))

                    pbar.update(task, advance=1,
                                details=f'{ax} / {name} in {pbar.get_time() - ti:.2f} s')

            pbar.update(task, advance=0,
                        details=f'Total : {pbar.tasks[0].finished_time:.2f} s')
            pbar.refresh()

        [setattr(self, f'{ax}domains', d) for ax, d in zip('xyz', self._domains)]
        self.cdomains = min(self._domains, key=len)

    def raw_show(self, domains=False, obstacles=False):

        if not hasattr(self, '_mask'):
            print('Free must be False')
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.colors import ListedColormap, BoundaryNorm

        nodes = [-11, 0, 1, 11]
        colors = ["mistyrose", "black", "white", "paleturquoise"]

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(nodes, len(colors) - 1)

        fig, axs = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True)
        for i in range(self.ndim):
            axs[i].imshow(self._mask[..., i].T, origin='lower', cmap=cmap, norm=norm)
            if obstacles:
                for obs in self.obstacles:
                    patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='r', fill=False, linewidth=3)
                    axs[i].add_patch(patch)
                    rx, ry = patch.get_xy()
                    cx = rx + patch.get_width()/2.0
                    cy = ry + patch.get_height()/2.0
                    msg = f'{obs.sid}\n{obs.description}'
                    axs[i].annotate(msg, (cx, cy), color='white', weight='bold',
                            fontsize=12, ha='center', va='center')

        if domains:
            for obs in self.xdomains:
                patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='b', fill=False, hatch='/', linewidth=3)
                axs[0].add_patch(patch)
                idx = _np.array((self._mask[..., 0] == -11).nonzero(), dtype=_np.int16).T
                axs[0].plot(*idx.T, 'ro')

            for obs in self.ydomains:
                patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='b', fill=False, hatch='/', linewidth=3)
                axs[1].add_patch(patch)

        plt.show()

    def show(self, obstacles=True, domains=False, bounds=True, only_mesh=True):
        """ Plot 3d representation of computation domain. """
        viewer = _graphics.CDViewer(self)
        viewer.show(obstacles=obstacles, domains=domains, bounds=bounds, only_mesh=only_mesh)

    def _free(self):
        try:
            del self._mask
        except AttributeError:
            pass


if __name__ == "__main__":

    from .templates import TestCases

    # Geometry
    nx, ny, nz = 512, 512, 512
    shape, stencil = (nx, ny, nz), 3

    test_cases = TestCases(shape, stencil)
    cp = ComputationDomains(shape, test_cases.case9, stencil=stencil)
    cp.show(obstacles=True, domains=True, bounds=False)
