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
from nsfds3.graphics import CDViewer
from libfds.cutils import nonzeros, where


class ComputationDomains:
    """ Divide computation domain in several Subdomains based on obstacles in presence.

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with three int objects.
    obstacles : list, :py:class:`nsfds3.mesher.geometry.ObstacleSet`, optional
        Obstacles in the computation domain.
    bc : {'[ARZPW][ARZPW][ARZPW][ARZPW]'}, optional
        Boundary conditions. Must be a 6 characters string corresponding to
        left, right, front, back, bottom, and top boundaries respectively
    stencil : int, optional
        Size of the finite difference stencil.
    Npml : int, optional
        Number of points of the absorbing area (used if 'A' in `bc`).
    flat : tuple, list
        Consider a 2d computation domain. flat must be of the form (axis, index).

    TODO
    ----
        - overlapped => check if overlap with more that `stencil` points
        - with this new formulation, obstacles could be nested
        - optimizations :
            - domains.argwhere((0, 0, 0))
            - _np.unique(domains._mask, axis=3)
            - locations_to_cuboids
    """

    def __init__(self, shape, obstacles=None, bc='WWWWWW', stencil=3, Npml=15):
        self.shape = shape
        self.bc = bc.upper()
        self.stencil = stencil
        self._midstencil = int((stencil - 1) / 2)
        self.Npml = Npml
        self.volumic = len(shape) == 3

        if isinstance(obstacles, ObstacleSet):
            self.obstacles = obstacles
        else:
            self.obstacles = ObstacleSet(shape, obstacles, stencil=stencil)
        self.bounds = Domain(origin=(0, ) * len(shape), size=shape, bc=self.bc).faces

        # Init listing of configs encountered in the domain and uncentered domains list
        self.configs = set()
        self.domains = []

        # Init mask with obstacle locations
        self._mask = _np.zeros(shape + (3, ), dtype=_np.int8)
        self._cmask = _np.ones(shape, dtype=_np.int8)
        for obs in self.obstacles:
            self._mask[obs.slices + (slice(0, 3), )] = 1
            self._cmask[obs.slices] = False

        # List of bc
        self._bc_uncenter = ['W', 'Z', 'V']
        self._bc_center = ['P', ]

        # Fill mask
        self._fill_faces_tangent()
        self._fill_boundaries()
        self._fill_faces_normal()

        # Check if mask is ok
        #self._check_mask()

        # Find computation domains
        self._find_domains()

    def _find_domains(self):
        """ Find uncentered and centered domains. """
        schemes = Schemes(stencil=self.stencil, ndim=len(self.shape))
        configs = schemes.all

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

    def _update_domains(self, config):
        cuboids = locations_to_cuboids(self.argwhere(config))
        for cub in cuboids:
            self.domains.append(Domain(cub['origin'], cub['size'], tag=config))

        self.cdomains = DomainSet(shape=self.shape, subs=self.domains, stencil=self.stencil)

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

    def _fill_faces_tangent(self):
        """ Setup face for tangential schemes. """
        # Fill faces
        faces = [face for face in self.obstacles.faces if not face.clamped]
        for face in faces:
            self._mask[face.slices + (face.not_axis, )] = 0
            self._cmask[face.slices] = True

    def _fill_boundaries(self):
        """ Setup boundary areas. """
        # Fill boundaries
        if self.bc[0] in self._bc_uncenter:
            slices = (slice(None, self._midstencil), slice(None), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._mask[slices + (0, )] = self.stencil
            self._cmask[slices] = False
        if self.bc[1] in self._bc_uncenter:
            slices = (slice(-self._midstencil, None), slice(None), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._mask[slices + (0, )] = -self.stencil
            self._cmask[slices] = False
        if self.bc[2] in self._bc_uncenter:
            slices = (slice(None), slice(None, self._midstencil), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._mask[slices + (1, )] = self.stencil
            self._cmask[slices] = False
        if self.bc[3] in self._bc_uncenter:
            slices = (slice(None), slice(-self._midstencil, None), slice(None))
            if not self.volumic:
                slices = slices[:2]
            self._mask[slices + (1, )] = -self.stencil
            self._cmask[slices] = False
        if self.volumic:
            if self.bc[4] in self._bc_uncenter:
                slices = (slice(None), slice(None), slice(None, self._midstencil))
                self._mask[slices + (2, )] = self.stencil
                self._cmask[slices] = False
            if self.bc[5] in self._bc_uncenter:
                slices = (slice(None), slice(None), slice(-self._midstencil, None))
                self._mask[slices + (2,)] = -self.stencil
                self._cmask[slices] = False

        # Fix boundaries
        faces = [(bound, face) for bound, face in _it.product(self.bounds, self.obstacles.faces)
                 if bound.side == face.side and face.clamped]

        for bound, face in faces:
            slices = list(self._inner_slices(face))
            if bound.normal == 1:
                slices[face.axis] = slice(None, self._midstencil)
            else:
                slices[face.axis] = slice(-self._midstencil, None)
            self._mask[tuple(slices) + (slice(None, None), )] = 1
            self._cmask[tuple(slices)] = False

    def _fill_faces_normal(self):
        """ Setup faces for normal schemes. """
        # Fill faces
        faces = [face for face in self.obstacles.faces if not face.clamped]
        for face in faces:
            slices = list(self._inner_slices(face))
            if face.normal == 1:
                slices[face.axis] = slice(face.loc, face.loc + self._midstencil)
            else:
                slices[face.axis] = slice(face.loc - self._midstencil + 1, face.loc + 1)
            self._mask[tuple(slices) + (face.axis,)] = face.normal * self.stencil
            self._cmask[tuple(slices)] = False

        # Fix faces that overlap
        for face1, face2 in self.obstacles.faces.overlapped:
            inter = face1.intersection(face2)
            inter = [slice(min(i) + 1, max(i)) for i in inter]
            inter[face1.axis] = slice(inter[face1.axis].start - self._midstencil,
                                      inter[face1.axis].start + self._midstencil - 1)
            inter = tuple(inter)
            self._mask[inter + ((0, 1, 2), )] = 1
            self._cmask[inter] = False

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
            self._mask[tuple(slices) + (face1.axis, )] = face1.normal * self.stencil
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
            #idx = tuple(a + o for a, o in zip(self._argwhere(self._mask[slices], pattern), offset))
            mask_c = _np.ascontiguousarray(self._mask[slices])
            idx = tuple(a + o for a, o in zip(where(mask_c, pattern).T, offset))
            pattern_idx = [_np.concatenate((pattern_idx[i], idx[i])) for i in range(len(pattern))]

        #return unique(_np.array(pattern_idx, dtype=_np.int16).T)
        return unique(_np.array(pattern_idx).T)

    def show(self, axis=0, obstacles=True, mask=False, domains=False, bounds=True):
        """ Plot 3d representation of computation domain. """
        viewer = CDViewer(self)
        viewer.show(axis=axis, obstacles=obstacles, mask=mask, domains=domains, bounds=bounds)

    def zoom(self, ix=None, iy=None, iz=None):
        """ Plot a zoom at (ix, iy, iz) of the mask. """
        zoomer = CDViewer(self)
        zoomer.zoom(ix, iy, iz)

    def get_traces(self, axis=0, obstacles=True, mask=False, domains=False, bounds=True):
        """ Get only traces. """
        viewer = CDViewer(self)
        return viewer.get_traces(axis=axis, obstacles=obstacles,
                                 mask=mask, domains=domains, bounds=bounds)


if __name__ == "__main__":

    from .templates import TestCases

    # Geometry
    nx, ny, nz = 512, 512, 512
    shape, stencil = (nx, ny, nz), 3

    test_cases = TestCases(shape, stencil)
    cdomains = ComputationDomains(shape, test_cases.case9, stencil=stencil)
    cdomains.show(axis=0, obstacles=True, mask=False, domains=True, bounds=False)
    cdomains.zoom(ix=slice(None, None), iy=17, iz=slice(None, None))
