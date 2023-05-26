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
            - domains.argwhere((0, 0, 0))
            - _np.unique(domains._mask, axis=3)
            - locations_to_cuboids
    """

    _BC_U = ['W', 'A']
    _BC_C = ['P', ]

    def __init__(self, shape, obstacles=None, bc='WWWWWW', nbz=20, stencil=11, free=True):

        self.shape = shape
        self.ndim = len(shape)
        self.obstacles = obstacles
        self.bc = bc
        self.nbz = nbz
        self.stencil, self._midstencil = stencil, int((stencil - 1) / 2)

        if isinstance(obstacles, ObstacleSet):
            self.obstacles = obstacles
        else:
            self.obstacles = ObstacleSet(shape, obstacles, stencil=stencil)

        self.bounds = self.obstacles.bounds
        self.buffer = Domain(**buffer_kwargs(self.bc, self.nbz, self.shape))
        self.domains = []

        # Find computation domains based on mask values
        self.find_domains()

        # Free memory
        if free:
            self._free()

    def _mask_init(self):
        """
        Initialize a mask that will contain the following values :
            - 0 : obstacle
            - 1 : centered
            - +11/-11 : uncentered
        """
        self._umask = _np.ones(self.shape + (self.ndim, ), dtype=_np.int8)

        for obs in self.obstacles:
            self._umask[obs.sin + (slice(0, self.ndim), )] = 0

        # Fix covered faces to 1
        for f in self.obstacles.covered:
            self._umask[f.sin + (slice(0, self.ndim), )] = 0

        # Fix junction between objects that overlap face to face
        combs = [(f1, f2) for f1, f2 in _it.combinations(self.obstacles.faces, r=2) if f1.sid != f2.sid]
        for f1, f2 in combs:
            if f1.intersects(f2) and f1.side == f2.opposite:
                s = tuple(zip(*f1.inner_indices().intersection(f2.inner_indices())))
                self._umask[s + (slice(0, self.ndim), )] = 0

        # Fix junction between face to face periodic objects
        # TODO

    def _mask_setup(self):
        """
        Fill the mask according to finite difference scheme to be applied on each point.
        """

        self._mask_init()

        for f in self.obstacles.uncentered:
            fbox = f.box(self._midstencil)
            sn = fbox.periodic_slice(self.stencil) if f.periodic else fbox.sn
            base = _np.zeros(fbox.size, dtype=_np.int8)
            base[(slice(None), ) * self.ndim] = self._umask[f.base_slice + (f.axis, )] == 0
            base[self._umask[sn + (f.axis, )] == 1] *= f.normal * self.stencil
            base[base == 0] = 1
            self._umask[sn + (f.axis, )] *= base

        for f in [b for b in self.bounds if b.bc in self._BC_U]:
            sn = f.box(self._midstencil).sn
            self._umask[sn + (f.axis, )] = f.normal * self.stencil

            for o in f.overlapped:
                sn = o.inner_slices(f.axis)
                self._umask[sn + (f.axis, )] = 0

    def find_domains(self):

        # Fill masks
        ti = time.perf_counter()
        self._mask_setup()
        print(f'Masks filling     : done in {time.perf_counter() - ti:.4f} s')

        domains = [[] for i in range(self.ndim)]

        for n in range(self.ndim):

            for c in [1, self.stencil, -self.stencil]:

                idx = _np.array((self._umask[:, :, n] == c).nonzero(), dtype=_np.int16).T
                cuboids = locations_to_cuboids(_np.ascontiguousarray(idx))
                for cub in cuboids:
                    domains[n].append(Domain(cub['origin'], cub['size'], self.shape, tag=c))

        [setattr(self, f'{ax}domains', d) for ax, d in zip('xyz', domains)]
        self.cdomains = min(domains, key=len)

        print(f'Domains searching : done in {time.perf_counter() - ti:.4f} s')


    def raw_show(self, domains=False):

        if not hasattr(self, '_umask'):
            print('Free must be False')
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, axs = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True)
        for i in range(2):
            axs[i].imshow(self._umask[:, :, i].T, origin='lower', vmin=-2, vmax=2)
            for obs in self.obstacles:
                patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='r', fill=False, linewidth=3)
                axs[i].add_patch(patch)
                rx, ry = patch.get_xy()
                cx = rx + patch.get_width()/2.0
                cy = ry + patch.get_height()/2.0
                msg = f'{obs.sid}\n{obs.description}'
                axs[i].annotate(msg, (cx, cy), color='k', weight='bold',
                        fontsize=12, ha='center', va='center')

        if domains:
            for obs in self.xdomains:
                patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='k', fill=False, linewidth=3)
                axs[0].add_patch(patch)

            for obs in self.ydomains:
                patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='k', fill=False, linewidth=3)
                axs[1].add_patch(patch)

        plt.show()

    def show(self, obstacles=True, domains=False, bounds=True, only_mesh=True):
        """ Plot 3d representation of computation domain. """
        viewer = _graphics.CDViewer(self)
        viewer.show(obstacles=obstacles, domains=domains, bounds=bounds, only_mesh=only_mesh)

    def _free(self):
        try:
            del self._umask
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
