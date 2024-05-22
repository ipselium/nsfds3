# -*- coding: utf-8 -*-
#cython: language_level=3
#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: initializedcheck=False
#
# Copyright © 2016-2019 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
#
# Creation Date : 2017-03-21 - 16:51:13
"""
-----------


@author: Cyril Desjouy
"""

import time
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt, fabs, log2, ceil


ctypedef (Py_ssize_t, Py_ssize_t) ituple
ctypedef double[2][2] ftuple
cdef double NAN = float("NaN")

cdef inline void flatten(double[:, ::1] points, Py_ssize_t[:, ::1] indexes, 
                         double[:, ::1] xp, double[:, ::1] yp) noexcept nogil:

    cdef Py_ssize_t nx = xp.shape[0]
    cdef Py_ssize_t ny = yp.shape[1]
    cdef Py_ssize_t ix, iy, idx

    with nogil:
        for ix in range(nx):
            for iy in range(ny):
                idx = ix * ny + iy
                points[idx, 0] = xp[ix, iy]
                points[idx, 1] = yp[ix, iy]
                indexes[idx, 0] = ix
                indexes[idx, 1] = iy


cdef inline (double, double) minmax(double[::1] arr) noexcept nogil:
    cdef double min = arr[0]
    cdef double max = arr[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t n = arr.shape[0]

    with nogil:
        for i in range(1, n):
            if arr[i] < min:
                min = arr[i]
            elif arr[i] > max:
                max = arr[i]
    return min, max


cdef inline bint is_in_quadrilateral(double xi, double x1, double x2, double x3, double x4,
                                     double yi, double y1, double y2, double y3, double y4) noexcept nogil:

    if (xi - x1) * (y2 - y1) - (x2 - x1) * (yi - y1) < 0.:
        return False
    if (xi - x2) * (y3 - y2) - (x3 - x2) * (yi - y2) < 0.:
        return False
    if (xi - x3) * (y4 - y3) - (x4 - x3) * (yi - y3) < 0.:
        return False
    if (xi - x4) * (y1 - y4) - (x1 - x4) * (yi - y4) < 0.:
        return False
    return True


cdef inline ituple nearest_idx(double px, double py, BinaryGridIndex node) noexcept nogil:
    """Return the nearest neighbor of the provided point within the given BinaryGridIndex."""

    cdef double d
    cdef double _d = sqrt((node.data[0, 0] - px)**2 + (node.data[0, 1] - py)**2)
    cdef Py_ssize_t i
    cdef Py_ssize_t _i = 0
    cdef Py_ssize_t n = node.n

    for i in range(1, n):
        d = sqrt((node.data[i, 0] - px)**2. + (node.data[i, 1] - py)**2.)
        if d < _d:
            _d = d
            _i = i

    return node.indexes[_i, 0], node.indexes[_i, 1]


@cython.final
cdef class BinaryGridIndex:

    cdef readonly Py_ssize_t axis, leafsize, depth
    cdef Py_ssize_t n
    cdef readonly double umin, umax, vmin, vmax
    cdef readonly double median
    cdef readonly double[:, ::1] data
    cdef readonly Py_ssize_t[:, ::1] indexes
    cdef readonly BinaryGridIndex left_child, right_child
    cdef readonly bint final
    cdef readonly ftuple ranges

    def __cinit__(self, double[:, ::1] data, Py_ssize_t[:, ::1] indexes,
                        ftuple ranges, Py_ssize_t leafsize=32, Py_ssize_t depth = 1):

        self.depth = depth
        self.final = False

        self.data = data
        self.indexes = indexes
        self.n = self.data.shape[0]
        self.ranges = ranges
        self.umin, self.umax = self.ranges[0]
        self.vmin, self.vmax = self.ranges[1]

        self.axis = self.depth % 2
        # if self.umax - self.umin >= self.vmax - self.vmin:
        #     self.axis = 0
        # else:
        #     self.axis = 1

        self.median = 0.5 * (self.ranges[self.axis][0] + self.ranges[self.axis][1])

        self.left_child = None
        self.right_child = None
        self.leafsize = leafsize

        # max depth = int(log2(self.n / self.leafsize))
        if self.n >= 2. * self.leafsize and self.depth < 16:
            self.split()
        else:
            self.final = True

    cdef inline void split(self) noexcept:

        cdef Py_ssize_t i
        cdef list left = []
        cdef list right = []

        for i in range(self.n):
            if self.data[i, self.axis] <= self.median:
                left.append(i)
            if self.data[i, self.axis] >= self.median:
                right.append(i)

        if len(left) < self.leafsize or len(right) < self.leafsize:
            self.final = True
        else:
            self.split_data(left, right)

    cdef inline void split_data(self, list left, list right) noexcept:

        cdef Py_ssize_t i, idx
        cdef Py_ssize_t n_left = len(left)
        cdef Py_ssize_t n_right = len(right)
        cdef ftuple ranges_left, ranges_right
        cdef double[:, ::1] data_left = np.empty((n_left, 2))
        cdef double[:, ::1] data_right = np.empty((n_right, 2))
        cdef Py_ssize_t[:, ::1] indexes_left = np.empty((n_left, 2), dtype=int)
        cdef Py_ssize_t[:, ::1] indexes_right = np.empty((n_right, 2), dtype=int)

        for i in range(n_left):
            idx = left[i]
            data_left[i, 0] = self.data[idx, 0]
            data_left[i, 1] = self.data[idx, 1]
            indexes_left[i, 0] = self.indexes[idx, 0]
            indexes_left[i, 1] = self.indexes[idx, 1]

        for i in range(n_right):
            idx = right[i]
            data_right[i, 0] = self.data[idx, 0]
            data_right[i, 1] = self.data[idx, 1]
            indexes_right[i, 0] = self.indexes[idx, 0]
            indexes_right[i, 1] = self.indexes[idx, 1]

        if self.axis == 0:
            ranges_left = [[self.umin, self.median], [self.vmin, self.vmax]]
            ranges_right = [[self.median, self.umax], [self.vmin, self.vmax]]
        else:
            ranges_left = [[self.umin, self.umax], [self.vmin, self.median]]
            ranges_right = [[self.umin, self.umax], [self.median, self.vmax]]

        self.left_child = BinaryGridIndex(data_left, indexes_left, ranges_left,
                                          leafsize=self.leafsize, depth=self.depth+1)
        self.right_child = BinaryGridIndex(data_right, indexes_right, ranges_right,
                                           leafsize=self.leafsize, depth=self.depth+1)
        self.data = np.zeros((0, 0))

    cpdef list get_nodes(self):
        """Breadth First Search algorithm to list all nodes.

        Reference
        ---------

        https://stackoverflow.com/questions/22752471/how-to-iterate-through-a-binary-search-tree-in-pythonno-recursion
        """

        cdef list queue = [self]
        cdef list result = []
        cdef BinaryGridIndex node

        while queue:
            node = queue.pop(0)
            result.append(node)

            if node.left_child is not None:
                queue.append(node.left_child)
            if node.right_child is not None:
                queue.append(node.right_child)

        return result

    cdef inline Py_ssize_t contains(self, double vx, double vy) noexcept:
        """Report wether Binary Index contains point of coordinates (vx, vy)."""
        if vx <= self.umax:
            if vx >= self.umin:
                if vy <= self.vmax:
                    if vy >= self.vmin:
                        return 1
        return 0

    def __str__(self):
        s = f'ranges : {self.umin} - {self.umax} / {self.vmin} - {self.vmax}\n'
        s += f'axis : {self.axis}\n'
        s += f'median : {self.median}\n'
        return s

    def __repr__(self):
        return str(self)


@cython.final
cdef class GridInterpolation:
    """Interpolate data from irregular grid.

    Notes
    -----
    - 2 to 16 times faster for mapping creation (self.mapping) than scipy.spatial.Delauney
    - 50 times faster for neighbor searching (self.query) than scipy.spatial.KDTree
    - 2 times faster for interpolation (self.interp) than scipy.inteprolate.griddata
    """

    cdef double[:, ::1] xp, yp
    cdef double[:, ::1] xp_scale, yp_scale
    cdef double[::1] xi, yi
    cdef Py_ssize_t nx, ny, nx_new, ny_new
    cdef readonly double xscale, yscale
    cdef readonly Py_ssize_t[:, ::1] mapping
    cdef readonly BinaryGridIndex tree
    cdef readonly Py_ssize_t extend
    cdef bint log

    def __cinit__(self, double[:, ::1] xp, double[:, ::1] yp,
                        double[::1] xi, double[::1] yi, Py_ssize_t leafsize = 16, bint log=False):

        cdef Py_ssize_t[:, ::1] indexes
        cdef double[:, ::1] points
        cdef double umin, umax, vmin, vmax, ti
        cdef double dxmax, dymax
        cdef Py_ssize_t ix, iy
        cdef ftuple ranges

        self.log = log

        # Physical grid
        self.xp = xp
        self.yp = yp
        self.nx = self.xp.shape[0]
        self.ny = self.yp.shape[1]

        # Regular grid
        self.xi = xi
        self.yi = yi
        self.nx_new = self.xi.shape[0]
        self.ny_new = self.yi.shape[0]

        # Mapping
        self.mapping = np.empty((self.nx_new * self.ny_new, 2), dtype=int)

        # Rescale data
        if log:
            ti = time.perf_counter()

        self.xp_scale = np.empty((self.nx, self.ny))
        self.yp_scale = np.empty((self.nx, self.ny))
        self.xscale = 1. / (np.array(self.xp).max() - np.array(self.xp).min() * self.ny / self.nx)
        self.yscale = 1. / (np.array(self.yp).max() - np.array(self.yp).min() * self.nx / self.ny)
        for ix in range(self.nx):
            for iy in range(self.ny):
                self.xp_scale[ix, iy] = self.xp[ix, iy] * self.xscale
                self.yp_scale[ix, iy] = self.yp[ix, iy] * self.yscale

        if log:
            print(f'Scale data in {(time.perf_counter() - ti) * 1e6} us')
            ti = time.perf_counter()

        # Create Binary index of physical grid
        points = np.empty((self.nx * self.ny, 2))
        indexes = np.empty((self.nx * self.ny, 2), dtype=int)
        flatten(points, indexes, self.xp_scale, self.yp_scale)
        umin, umax = minmax(np.ascontiguousarray(points[:, 0]))
        vmin, vmax = minmax(np.ascontiguousarray(points[:, 1]))
        ranges = [[umin, umax], [vmin, vmax]]
        dxmax = np.diff(xp, axis=0).max()
        dymax = np.diff(yp, axis=1).max()
        self.extend = 11 # max(5, int(ceil(max(dxmax, dymax) / min(dxmax, dymax)) // 2 * 2 + 1))

        if log:
            print(f'Prepare data in {(time.perf_counter() - ti) * 1e6} us')
            ti = time.perf_counter()

        self.tree = BinaryGridIndex(points, indexes, ranges, leafsize=leafsize)

        if log:
            print(f'BinaryIndex generated in {(time.perf_counter() - ti) * 1e3} ms')
            ti = time.perf_counter()

        self.query_grid()

        if log:
            print(f'Grid mapping generated in {(time.perf_counter() - ti) * 1e3} ms')

    cpdef double[:, ::1] interp2d(self, double[:, ::1] p, Py_ssize_t deg=2, Py_ssize_t cpu=2) noexcept:
        """Interpolate p using Shepard's Inverse distance weighting."""
        assert p.shape[0] == self.nx and p.shape[1] == self.ny, "wrong size"
        assert cpu > 0, "Wrong cpu number"
        return self._interp_shepard(p, deg, cpu)

    cdef double[:, ::1] _interp_shepard(self, double[:, ::1] p, Py_ssize_t deg, Py_ssize_t cpu) noexcept:

        cdef double[:, ::1] out = np.zeros((self.nx_new, self.ny_new))
        cdef double pdn, dn, ds
        cdef Py_ssize_t i, ix, iy, a, b, _ix, _iy
        cdef int[4][2] idx

        for ix in prange(self.nx_new, nogil=True, num_threads=cpu):
            for iy in range(self.ny_new):
                i = ix * self.ny_new + iy
                _ix = self.mapping[i, 0]
                _iy = self.mapping[i, 1]
                if _ix == -1:
                    out[ix, iy] = NAN
                    continue
                idx = [[_ix, _iy], [_ix + 1, _iy], [_ix + 1, _iy + 1], [_ix, _iy + 1]]
                ds = 0.
                pdn = 0.
                for i in range(4):
                    a = idx[i][0]
                    b = idx[i][1]
                    dn = 1 / sqrt((self.xp[a, b] - self.xi[ix])**2 + (self.yp[a, b] - self.yi[iy])**2) ** deg
                    ds = ds + dn
                    pdn = pdn + p[a, b] * dn
                out[ix, iy] = pdn / ds
        return out

    cpdef inline ituple query(self, double px, double py) noexcept:
        """Return a (close) neighbor of the point of coordinates (px, py).
        If no neighbor is found, returns -1

        Note
        ----
        The returned neighbor will not necessarily be the nearest neighbor.
        However, it is the nearest neighbor in the branch in which the point
        (px, py) is located.
        For a closer neighbor, one can use query_cell that return the bottom-left
        point of the quadrilateral in which is located the point (px, py)
        """

        cdef BinaryGridIndex node

        px *= self.xscale
        py *= self.yscale
        node = self.query_node(px, py)

        return nearest_idx(px, py, node)

    cpdef inline ituple query_cell(self, double px, double py) noexcept:
        """Return the bottom-left point of the quadrilateral in which is located
        the point of coordinates (px, py). If the point does not belong to any
        quadrilateral, return (-1, -1)
        """

        cdef BinaryGridIndex node
        cdef Py_ssize_t ix = - 1, iy = - 1
        cdef Py_ssize_t _ix, _iy

        px *= self.xscale
        py *= self.yscale
        node = self.query_node(px, py)

        if not node.contains(px, py):
            return ix, iy

        _ix, _iy = nearest_idx(px, py, node)
        ix, iy = self._query_single_point(px, py, _ix, _iy)
        if ix != -1:
            return ix, iy

        ix, iy = self._query_surrounding_points(px, py, _ix, _iy)
        if ix != -1:
            return ix, iy

        return ix, iy

    cdef inline void query_grid(self) noexcept:

        cdef Py_ssize_t i, ix, iy, iu, iv

        for ix in range(self.nx_new):
            for iy in range(self.ny_new):
                i = ix * self.ny_new + iy
                iu, iv = self.query_cell(self.xi[ix], self.yi[iy])
                self.mapping[i, 0] = iu
                self.mapping[i, 1] = iv

    cdef inline BinaryGridIndex query_node(self, double px, double py) noexcept:
        """Return the node containing the provided point."""

        cdef BinaryGridIndex node = self.tree

        while not node.final:
            if node.axis == 0:
                if px <= node.median:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                if py <= node.median:
                    node = node.left_child
                else:
                    node = node.right_child
        return node

    cdef inline ituple _query_surrounding_points(self, double px, double py, Py_ssize_t nix, Py_ssize_t niy) noexcept:
        """Return the bottom left grid point of the cell containing the provided point using points surrounding the neighbor."""

        cdef Py_ssize_t ix, iy, i
        cdef Py_ssize_t _ix = -1, _iy = -1

        for i in range(2, self.extend, 2):
            for ix in range(-i, i+1, 2):
                for iy in range(-i, i+1, 2):
                    if not (fabs(ix) < i and  fabs(iy) < i):
                        _ix, _iy = self._query_single_point(px, py, nix + ix, niy + iy)
                        if _ix != -1:
                            return _ix, _iy
        return _ix, _iy

    cdef inline ituple _query_single_point(self, double xi, double yi, Py_ssize_t ix, Py_ssize_t iy) noexcept:
        """Return the bottom left grid point of the cell containing the point using the first neighbor.

        Note
        ----
        This method search for the point in the 4 cells surrounding the neighbor.
        """

        cdef double x1, x2, x3, x4, y1, y2, y3, y4

        if 0 <= ix < self.nx:
            x1 = self.xp_scale[ix, iy]
        else:
            return -1, -1
        if 0 <= iy < self.ny:
            y1 = self.yp_scale[ix, iy]
        else:
            return -1, -1

        if ix + 1 < self.nx and iy + 1 < self.ny:

            x2 = self.xp_scale[ix, iy+1]
            y2 = self.yp_scale[ix, iy+1]
            x3 = self.xp_scale[ix+1, iy+1]
            y3 = self.yp_scale[ix+1, iy+1]
            x4 = self.xp_scale[ix+1, iy]
            y4 = self.yp_scale[ix+1, iy]

            if is_in_quadrilateral(xi, x1, x2, x3, x4, yi, y1, y2, y3, y4):
                return (ix, iy)

        if ix + 1 < self.nx and iy - 1 >= 0:

            x2 = self.xp_scale[ix+1, iy]
            y2 = self.yp_scale[ix+1, iy]
            x3 = self.xp_scale[ix+1, iy-1]
            y3 = self.yp_scale[ix+1, iy-1]
            x4 = self.xp_scale[ix, iy-1]
            y4 = self.yp_scale[ix, iy-1]

            if is_in_quadrilateral(xi, x1, x2, x3, x4, yi, y1, y2, y3, y4):
                return (ix, iy-1)

        if ix - 1 >= 0 and iy + 1 < self.ny:

            x2 = self.xp_scale[ix-1, iy]
            y2 = self.yp_scale[ix-1, iy]
            x3 = self.xp_scale[ix-1, iy+1]
            y3 = self.yp_scale[ix-1, iy+1]
            x4 = self.xp_scale[ix, iy+1]
            y4 = self.yp_scale[ix, iy+1]

            if is_in_quadrilateral(xi, x1, x2, x3, x4, yi, y1, y2, y3, y4):
                return (ix-1, iy)

        if ix - 1 >= 0 and iy - 1 >= 0:

            x2 = self.xp_scale[ix, iy-1]
            y2 = self.yp_scale[ix, iy-1]
            x3 = self.xp_scale[ix-1, iy-1]
            y3 = self.yp_scale[ix-1, iy-1]
            x4 = self.xp_scale[ix-1, iy]
            y4 = self.yp_scale[ix-1, iy]

            if is_in_quadrilateral(xi, x1, x2, x3, x4, yi, y1, y2, y3, y4):
                return (ix-1, iy-1)

        return -1, -1

    def __str__(self):
        nodes = [node for node in self.tree.get_nodes() if node.final]
        s = ''
        s += f'Final Leaves   : {len(nodes)}\n'
        s += f'Max. depth     : {max(node.depth for node in nodes)}\n'
        s += f'Min. Leaf Size : {min(np.array(node.data).shape[0] for node in nodes)}\n'
        s += f'Max. Leaf Size : {max(np.array(node.data).shape[0] for node in nodes)}\n'
        s += f'Extend search  : {self.extend}\n'
        return s

    def __repr__(self):
        return str(self)