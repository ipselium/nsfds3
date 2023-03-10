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
from ._geometry import ObstacleSet, DomainSet, Domain
from nsfds3.utils.misc import locations_to_cuboids, unique, Schemes, scheme_to_str
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


    def __init__(self, shape, obstacles=None, bc='WWWWWW', stencil=11, free=True):
        self.shape = shape
        self.bc = bc.upper()
        self.stencil = stencil
        self._midstencil = int((stencil - 1) / 2)
        self.volumic = len(shape) == 3

        if isinstance(obstacles, ObstacleSet):
            self.obstacles = obstacles
        else:
            self.obstacles = ObstacleSet(shape, obstacles, stencil=stencil)

        self.bounds = Domain(origin=(0, ) * len(shape), size=shape, bc=self.bc).faces

        # Init listing of domains
        self.domains = []

        # Init mask with obstacle locations
        # umask will be filled with 0 if centered scheme and \pm stencil if uncentered
        # cmask values are true when scheme is centered
        self._umask = _np.zeros(shape + (3, ), dtype=_np.int8)
        self._cmask = _np.ones(shape, dtype=_np.int8)
        for obs in self.obstacles:
            self._umask[obs.slices + (slice(0, 3), )] = 1
            self._cmask[obs.slices] = False

        # Fill mask
        self._fill_faces_tangent()
        self._fill_boundaries()
        self._fill_faces_normal()

        # Find computation domains
        self._find_domains()

        # Free memory
        if free:
            self._free()

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

    def _inner_slices(self, f):
        """ Return inner slices of a face. """

        xmin = f.ix[0] if f.ix[0] == 0 or (0, 1) in f.bounded else f.ix[0] + 1
        xmax = f.ix[1] + 1 if f.ix[1] == self.shape[0] - 1 or (0, -1) in f.bounded else f.ix[1]
        ymin = f.iy[0] if f.iy[0] == 0 or (1, 1) in f.bounded else f.iy[0] + 1
        ymax = f.iy[1] + 1 if f.iy[1] == self.shape[1] - 1 or (1, -1) in f.bounded else f.iy[1]
        if self.volumic:
            zmin = f.iz[0] if f.iz[0] == 0 or (2, 1) in f.bounded else f.iz[0] + 1
            zmax = f.iz[1] + 1 if f.iz[1] == self.shape[2] - 1 or (2, -1) in f.bounded else f.iz[1]
            slices = [slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)]
        else:
            slices = [slice(xmin, xmax), slice(ymin, ymax)]

        slices[f.axis] = f.slices[f.axis]

        return slices

    def _fill_boundaries(self):
        """ Setup boundary areas. """
        # Fill boundaries
        if self.bc[0] in self._BC_U:
            slices = (slice(None, self._midstencil), slice(None), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._umask[slices + (0, )] = self.stencil
            self._cmask[slices] = False
        if self.bc[1] in self._BC_U:
            slices = (slice(-self._midstencil, None), slice(None), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._umask[slices + (0, )] = -self.stencil
            self._cmask[slices] = False
        if self.bc[2] in self._BC_U:
            slices = (slice(None), slice(None, self._midstencil), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._umask[slices + (1, )] = self.stencil
            self._cmask[slices] = False
        if self.bc[3] in self._BC_U:
            slices = (slice(None), slice(-self._midstencil, None), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._umask[slices + (1, )] = -self.stencil
            self._cmask[slices] = False
        if self.volumic:
            if self.bc[4] in self._BC_U:
                slices = (slice(None), slice(None), slice(None, self._midstencil))
                self._umask[slices + (2, )] = self.stencil
                self._cmask[slices] = False
            if self.bc[5] in self._BC_U:
                slices = (slice(None), slice(None), slice(-self._midstencil, None))
                self._umask[slices + (2,)] = -self.stencil
                self._cmask[slices] = False

        # Fix boundaries that are a superset of an obstacle face
        fix = [(bound, face) for bound, face in _it.product(self.bounds, self.obstacles.faces)
                 if bound.side == face.side and face.clamped]

        for bound, face in fix:
            slices = list(self._inner_slices(face))
            if bound.normal == 1:
                slices[face.axis] = slice(None, self._midstencil)
            else:
                slices[face.axis] = slice(-self._midstencil, None)
            self._umask[tuple(slices) + (slice(None, None), )] = 1
            self._cmask[tuple(slices)] = False

    def _fill_faces_tangent(self):
        """ Setup face for tangential schemes. """
        # Fill faces that are not clamped
        for face in self.obstacles.faces.not_clamped:
            self._umask[face.slices + (face.not_axis, )] = 0
            self._cmask[face.slices] = True

    def _fill_faces_normal(self):
        """ Setup faces for normal schemes. """
        # Fill faces that are not clamped
        for face in self.obstacles.faces.not_clamped:
            slices = list(self._inner_slices(face))
            if face.normal == 1:
                slices[face.axis] = slice(face.loc, face.loc + self._midstencil)
            else:
                slices[face.axis] = slice(face.loc - self._midstencil + 1, face.loc + 1)
            self._umask[tuple(slices) + (face.axis,)] = face.normal * self.stencil
            self._cmask[tuple(slices)] = False

        # Fix faces that overlap
        for face1, face2 in self.obstacles.faces.overlapped:
            inter = face1.intersection(face2)
            #inter = [slice(min(i) + 1, max(i)) for i in inter]
            inter = [slice(min(i) + 1 if min(i) != max(i) else min(i), max(i)) for i in inter]
            inter[face1.axis] = slice(inter[face1.axis].start - self._midstencil,
                                      inter[face1.axis].start + self._midstencil - 1)
            self._umask[tuple(inter) + ((0, 1, 2), )] = 1
            self._cmask[tuple(inter)] = False

        # Fix faces that have common edges
        for face1, face2 in self.obstacles.faces.common_edge:
            inter = face1.intersection(face2)
            lengths = [len(c) for c in inter]
            axis = lengths.index(max(lengths))
            slices = [slice(min(c), max(c) + 1) for c in inter]
            if face1.normal == 1:
                slices[face1.axis] = slice(face1.loc, face1.loc + self._midstencil)
            else:
                slices[face1.axis] = slice(face1.loc - self._midstencil + 1, face1.loc + 1)
            slices[axis] = slice(min(inter[axis]) + 1, max(inter[axis]))
            self._umask[tuple(slices) + (face1.axis, )] = face1.normal * self.stencil
            self._cmask[tuple(slices)] = False

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
