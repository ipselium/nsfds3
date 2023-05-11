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
from .geometry import ObstacleSet, DomainSet, Domain
from nsfds3.utils.misc import locations_to_cuboids, unique, Schemes, scheme_to_str, buffer_kwargs
import nsfds3.graphics as _graphics
from libfds.cutils import nonzeros, where


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
    stencil : int, optional
        Size of the finite difference stencil.
    free: bool, optional
        Free memory after the domains are found

    TODO
    ----
        - overlapped => check if overlap with more that `stencil` points
        - with this new formulation, obstacles could be nested
        - optimizations :
            - domains.argwhere((0, 0, 0))
            - _np.unique(domains._mask, axis=3)
            - locations_to_cuboids
    """

    _BC_U = ['W', 'A']
    _BC_C = ['P', ]

    def __init__(self, shape, obstacles=None, bc='WWWWWW', nbz=20, stencil=11, free=True):

        self.shape = shape
        self.obstacles = obstacles
        self.bc = bc
        self.nbz = nbz
        self.stencil, self._midstencil = stencil, int((stencil - 1) / 2)

        if isinstance(obstacles, ObstacleSet):
            self.obstacles = obstacles
        else:
            self.obstacles = ObstacleSet(shape, obstacles, stencil=stencil)

        # Init mask with obstacle locations
        # umask will be filled with 0 if centered scheme and \pm stencil if uncentered
        # cmask values are true when scheme is centered
        self._umask = _np.zeros(shape + (3, ), dtype=_np.int8)
        self._cmask = _np.ones(shape, dtype=_np.int8)
        for obs in self.obstacles:
            self._umask[obs.sn + (slice(0, 3), )] = 1
            self._cmask[obs.sn] = False

        self.bounds = self.obstacles.bounds
        self.buffer = Domain(**buffer_kwargs(self.bc, self.nbz, self.shape))
        self.domains = []

        self._fill()

        # Find computation domains
        #self._find_domains()

        # Free memory
        if free:
            self._free()

    def _uncentered_edges(self, f):
        """ Return the edges of the uncentered area corresponding to the face f."""
        return f.box(self._midstencil).indices - f.box(self._midstencil).inner_indices(f.axis)

    def _fix_edges(self, f):
        """ Return the edges to be added to uncentered areas."""
        edges = self._uncentered_edges(f)
        fix = set()

        for o in f.colinear:
            fo = o.get_same_face(f)
            fix |= edges & fo.box(self._midstencil).indices

        for o in f.overlapped:
            fix |= edges & o.indices
            if f.intersects(o.get_opposite_face(f)):
                fix -= edges & o.edges_indices(f.axis)

        for b in self.bounds:
            exceptions = set()
            for o in f.overlapped:
                exceptions |= edges & o.indices
            fix |= (edges - exceptions) & b.indices

        return fix

    def _update_indices(self):
        """ Set indices of all uncentered areas.

        Procedure :

            * Initialize indices for all obstacle faces and inner indices for free faces
            * For overlapped face, take care about surrounding obstacles and bounds
            * For face that are colinear, take care about the connection and bounds
        """

        for f in self.obstacles.uncentered:

            # Inner indice of a box ahead of the object for all faces
            f.uncentered = f.box(self._midstencil).inner_indices(f.axis)

            # Add missing edges and boundaries
            f.uncentered |= self._fix_edges(f)

            # Remove overlapped areas
            for o in f.overlapped:
                f.uncentered -= o.inner_indices(f.axis)


        for f in [b for b in self.bounds if b.bc in self._BC_U]:

            f.uncentered = f.box(self._midstencil).indices

            for o in f.overlapped:
                f.uncentered -= o.inner_indices(f.axis)

            exception = [o1.indices & o2.indices for (o1, o2) in _it.combinations(f.overlapped, r=2)]
            f.uncentered -= set(_it.chain(*exception))

    def _check_indices(self):
        """ Check if uncentered areas overlap."""
        overlapped_indices = []
        for f1, f2 in self.obstacles.face_combination:
            if f1.uncentered & f2.uncentered:
                overlapped_indices.append([f1, f2])

        if overlapped_indices:
            msg = '\n'
            for f1, f2 in overlapped_indices:
                msg += f'{f1}\nintersects\n{f2}\n---\n'
            #raise ValueError(msg)
            #print(msg)

    def _fill(self):
        """ Fill masks using indices of uncentered areas. """
        self._update_indices()
        self._check_indices()

        for f in self.obstacles.uncentered + self.obstacles.bounds:
            sn = tuple(zip(*f.uncentered))
            if sn:
                self._umask[sn + (f.axis, )] = f.normal * self.stencil
                self._cmask[sn] = False

    def raw_show(self):

        if not hasattr(self, '_umask'):
            print('Free must be False')
            return 

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, axs = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True)
        for i in range(2):
            axs[i].imshow(self._umask[:, :, i].T, origin='lower')
            for obs in self.obstacles:
                patch = Rectangle(obs.origin, *[s - 1 for s in obs.size], color='r', fill=False, linewidth=3)
                axs[i].add_patch(patch)
                rx, ry = patch.get_xy()
                cx = rx + patch.get_width()/2.0
                cy = ry + patch.get_height()/2.0
                msg = f'{obs.sid}\n{obs.description}'
                axs[i].annotate(msg, (cx, cy), color='k', weight='bold',
                        fontsize=12, ha='center', va='center')
        plt.show()

    def show(self, obstacles=True, domains=False, bounds=True, only_mesh=True):
        """ Plot 3d representation of computation domain. """
        viewer = _graphics.CDViewer(self)
        viewer.show(obstacles=obstacles, domains=domains, bounds=bounds, only_mesh=only_mesh)


# Pas revu :

    def _find_domains(self):
        """ Find uncentered and centered domains. """
        configs = Schemes(stencil=self.stencil, ndim=len(self.shape))(all=True)

        with _rp.Progress(_rp.TextColumn("[bold blue]{task.description:<20}...", justify="right"),
                          _rp.BarColumn(bar_width=None),
                          _rp.TextColumn("[progress.percentage]{task.percentage:>3.1f}% •"),
                          _rp.TimeRemainingColumn(),
                          _rp.TextColumn("• {task.fields[details]}")) as pbar:

            task = pbar.add_task("[red]Building domains...",
                                 total=len(configs), details='Starting...')

            for c in configs:
                ti = pbar.get_time()
                self._update_domains(c)
                pbar.update(task, advance=1,
                            details=f'{scheme_to_str(c)} in {pbar.get_time() - ti:.2f} s')
            pbar.update(task, advance=0,
                        details=f'Total : {pbar.tasks[0].finished_time:.2f} s')
            pbar.refresh()

        self._update_domains_bc()
        self.domains = DomainSet(shape=self.shape, subs=self.domains, stencil=self.stencil)

    def _update_domains(self, config):
        idx = self.argwhere(config)
        if idx is not None:
            cuboids = locations_to_cuboids(idx)
            for cub in cuboids:
                self.domains.append(Domain(cub['origin'], cub['size'], tag=config))

    def _update_domains_bc(self):
        """
            Set P marker to periodic bounds.

            TODO : bc W à fixer également !
        """

        for sub in self.domains:
            if sub.scm_fx == 'c' and (sub.ix[0] == 0 or sub.ix[1] == self.shape[0] - 1):
                sub.scm_fx = 'P'
            if sub.scm_fy == 'c' and (sub.iy[0] == 0 or sub.iy[1] == self.shape[1] - 1):
                sub.scm_fy = 'P'
            if self.volumic:
                if sub.scm_fz == 'c' and (sub.iz[0] == 0 or sub.iz[1] == self.shape[2] - 1):
                    sub.scm_fz = 'P'

    @staticmethod
    def _argwhere(mask, pattern):
        """ Return a list of indices where mask equals pattern. """
        if len(pattern) == 2:
            px, py = pattern
            x = mask[:, :, 0]
            y = mask[:, :, 1]
            return ((x == px) & (y == py)).nonzero()

        px, py, pz = pattern
        x = mask[:, :, :, 0]
        y = mask[:, :, :, 1]
        z = mask[:, :, :, 2]
        return ((x == px) & (y == py) & (z == pz)).nonzero()

    def argwhere(self, pattern):
        """ Return a list of indices where mask equals pattern. """

        # Look at self._cmask to find centered schemes location
        if pattern in [(0, 0, 0), (0, 0)]:
            return nonzeros(self._cmask)   # cython alternative (x8)
            #return _np.array(self._cmask.nonzero(), dtype=_np.int16).T

        # Only look arround obstacles to find uncentered schemes location
        pattern_idx = [_np.array([], dtype=_np.int8) for _ in range(3)]
        for obs in _it.chain(self.obstacles, self.bounds):
            slices = obs.box(self.shape, stencil=self.stencil)
            offset = [s.start for s in slices]
            #idx = tuple(a + o for a, o in zip(self._argwhere(self._umask[slices], pattern), offset))
            mask_c = _np.ascontiguousarray(self._umask[slices])
            idx = tuple(a + o for a, o in zip(where(mask_c, pattern).T, offset))
            pattern_idx = [_np.concatenate((pattern_idx[i], idx[i])) for i in range(len(pattern))]

        if any([len(i) for i in pattern_idx]):
            #return unique(_np.array(pattern_idx, dtype=_np.int16).T)
            return unique(_np.array(pattern_idx).T)
        else:
            return None

    def show(self, obstacles=True, domains=False, bounds=True, only_mesh=True):
        """ Plot 3d representation of computation domain. """
        viewer = _graphics.CDViewer(self)
        viewer.show(obstacles=obstacles, domains=domains, bounds=bounds,
                    only_mesh=only_mesh)

    def _free(self):
        try:
            del self._umask
            del self._cmask
        except AttributeError:
            pass


if __name__ == "__main__":

    from .templates import TestCases

    # Geometry
    nx, ny, nz = 512, 512, 512
    shape, stencil = (nx, ny, nz), 3

    test_cases = TestCases(shape, stencil)
    cdomains = ComputationDomains(shape, test_cases.case9, stencil=stencil)
    cdomains.show(obstacles=True, domains=True, bounds=False)
