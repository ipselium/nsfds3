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
# This file is part of ofdlib2
#
# ofdlib2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ofdlib2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ofdlib2. If not, see <http://www.gnu.org/licenses/>.
#
#
# Creation Date : 2017-03-21 - 16:51:13
"""
-----------


@author: Cyril Desjouy
"""

import numpy as np
cimport numpy as np
cimport cython
from numpy cimport int64_t, uint8_t

ctypedef np.npy_int8 INT8


ctypedef fused integer:
    short
    long
    int
    INT8


cpdef INT8[:, ::1] is_equal(integer[:, ::1] array, int value):
    """ Return a mask where array == value. """

    cdef Py_ssize_t ix, iy
    cdef int nx = array.shape[0]
    cdef int ny = array.shape[1]
    cdef INT8[:, ::1] out = np.zeros((nx, ny), dtype=np.int8)

    for ix in range(nx):
        for iy in range(ny):
            if array[ix, iy] == value:
                out[ix, iy] = 1
    return out


cpdef list get_2d_cuboids(integer[:, ::1] mask, int ax=-1, int N=-1):
    """ Return a list of dictionnaries containing cuboid parameters.

    Parameters
    ----------
        mask : np.array
            Search cuboids in mask
        ax : int, optional
            Direction of the search. Can be 0 (x), 1 (y), or other (center)
        N : int, optional
            Fix one dimension of the output cuboids. Only for ax=0 or ax=1.
    """
    if N > 0 and ax == 0:
        return _get_x_2d_cuboids(mask, N)

    elif N > 0 and ax == 1:
        return _get_y_2d_cuboids(mask, N)

    else:
        return _get_2d_cuboids(mask)


cpdef list get_3d_cuboids(integer[:, :, ::1] mask, int ax=-1, int N=-1):
    """ Return a list of dictionnaries containing cuboid parameters.

    Parameters
    ----------
        mask : np.array
            Search cuboids in mask
        ax : int, optional
            Direction of the search. Can be 0 (x), 1 (y), 2 (z), or other (center)
        N : int, optional
            Fix one dimension of the output cuboids. Only for ax=0, ax=1, ax=2.
    """
    if N > 0 and ax == 0:
        return _get_x_3d_cuboids(mask, N)

    elif N > 0 and ax == 1:
        return _get_y_3d_cuboids(mask, N)

    elif N > 0 and ax == 2:
        return _get_z_3d_cuboids(mask, N)

    else:
        return _get_3d_cuboids(mask)


cdef list _get_2d_cuboids(integer[:, ::1] mask):

    cdef Py_ssize_t ix, iy
    cdef short cxi, cxf
    cdef short cyi, cyf
    cdef integer[:, ::1] m = mask.copy()
    cdef list cuboids = []

    while any_nonzero2d(m):

        cxi, cyi = first_nonzero2d(m)
        cxf, cyf = m.shape[0], m.shape[1]

        for iy in range(cyi, cyf):
            if m[cxi, iy] != 1:
                cyf = iy
                break

            for ix in range(cxi, cxf):
                if m[ix, iy] != 1:
                    cxf = ix
                    break

        if cxf - cxi > 0 and cyf - cyi > 0:
            cuboids.append(dict(origin=(cxi, cyi), size=(cxf - cxi, cyf - cyi)))
            for ix in range(cxi, cxf):
                for iy in range(cyi, cyf):
                    m[ix, iy] = 0
        else:
            break

    return cuboids


cdef list _get_3d_cuboids(integer[:, :, ::1] mask):

    cdef Py_ssize_t ix, iy, iz
    cdef short cxi, cxf
    cdef short cyi, cyf
    cdef integer[:, :, ::1] m = mask.copy()
    cdef list cuboids = []


    while any_nonzero3d(m):

        cxi, cyi, czi = first_nonzero3d(m)
        cxf, cyf, czf = m.shape[0], m.shape[1], m.shape[2]

        for iz in range(czi, czf):
            if m[cxi, cyi, iz] != 1:
                czf = iz
                break

            for iy in range(cyi, cyf):
                if m[cxi, iy, iz] != 1:
                    cyf = iy
                    break

                for ix in range(cxi, cxf):
                    if m[ix, iy, iz] != 1:
                        cxf = ix
                        break
        if cxf - cxi > 0 and cyf - cyi > 0 and czf - czi > 0:
            cuboids.append(dict(origin=(cxi, cyi, czi), size=(cxf - cxi, cyf - cyi, czf - czi)))
            for ix in range(cxi, cxf):
                for iy in range(cyi, cyf):
                    for iz in range(czi, czf):
                        m[ix, iy, iz] = 0
        else:
            break

    return cuboids


cdef list _get_x_2d_cuboids(integer[:, ::1] mask, int N):

    cdef Py_ssize_t iy
    cdef short cxi, cyi
    cdef short cyf, cyf_bis
    cdef integer[:, ::1] m = mask.copy()
    cdef list cuboids = []

    while any_nonzero2d(m):

        cxi, cyi = first_nonzero2d(m)
        cyf = m.shape[1]
        cyf_bis = m.shape[1]

        for iy in range(cyi, cyf):
            if m[cxi, iy] != 1:
                cyf = iy
                break

        for iy in range(cyi, cyf):
            if m[cxi + N - 1, iy] != 1:
                cyf_bis = iy
                break

        cyf = min(cyf, cyf_bis)

        if cyf - cyi > 0:
            cuboids.append(dict(origin=(cxi, cyi), size=(N, cyf - cyi)))
            for ix in range(cxi, cxi + N):
                for iy in range(cyi, cyf):
                    m[ix, iy] = 0
        else:
            break

    return cuboids


cdef list _get_x_3d_cuboids(integer[:, :, ::1] mask, int N):

    cdef Py_ssize_t iy, iz
    cdef short cxi, cyi, czi
    cdef short cyf, cyf_bis, czf, czf_bis
    cdef integer[:, :, ::1] m = mask.copy()
    cdef list cuboids = []

    while any_nonzero3d(m):

        cxi, cyi, czi = first_nonzero3d(m)
        cyf, czf = m.shape[1], m.shape[2]
        cyf_bis, czf_bis = m.shape[1], m.shape[2]

        for iz in range(czi, czf):
            if m[cxi, cyi, iz] != 1:
                czf = iz
                break

            for iy in range(cyi, cyf):
                if m[cxi, iy, iz] != 1:
                    cyf = iy
                    break

        for iz in range(czi, czf):
            if m[cxi + N - 1, cyi, iz] != 1:
                czf_bis = iz
                break

            for iy in range(cyi, cyf):
                if m[cxi + N - 1, iy, iz] != 1:
                    cyf_bis = iy
                    break

        cyf = min(cyf, cyf_bis)
        czf = min(czf, czf_bis)

        if cyf - cyi > 0 and czf - czi > 0:
            cuboids.append(dict(origin=(cxi, cyi, czi), size=(N, cyf - cyi, czf - czi)))
            for ix in range(cxi, cxi + N):
                for iy in range(cyi, cyf):
                    for iz in range(czi, czf):
                        m[ix, iy, iz] = 0
        else:
            break

    return cuboids


cdef list _get_y_2d_cuboids(integer[:, ::1] mask, int N):

    cdef Py_ssize_t ix
    cdef short cxi, cyi
    cdef short cxf, cxf_bis
    cdef integer[:, ::1] m = mask.copy()
    cdef list cuboids = []

    while any_nonzero2d(m):

        cxi, cyi = first_nonzero2d(m)
        cxf = m.shape[0]
        cxf_bis = m.shape[0]

        for ix in range(cxi, cxf):
            if m[ix, cyi] != 1:
                cxf = ix
                break

        for ix in range(cxi, cxf):
            if m[ix, cyi + N - 1] != 1:
                cxf_bis = ix
                break

        cxf = min(cxf, cxf_bis)

        if cxf - cxi > 0:
            cuboids.append(dict(origin=(cxi, cyi), size=(cxf - cxi, N)))
            for ix in range(cxi, cxf):
                for iy in range(cyi, cyi + N):
                    m[ix, iy] = 0
        else:
            break

    return cuboids


cdef list _get_y_3d_cuboids(integer[:, :, ::1] mask, int N):

    cdef Py_ssize_t ix, iz
    cdef short cxi, cyi, czi
    cdef short cxf, cxf_bis, czf, czf_bis
    cdef integer[:, :, ::1] m = mask.copy()
    cdef list cuboids = []

    while any_nonzero3d(m):

        cxi, cyi, czi = first_nonzero3d(m)
        cxf, czf = m.shape[0], m.shape[2]
        cxf_bis, czf_bis = m.shape[0], m.shape[2]

        for iz in range(czi, czf):
            if m[cxi, cyi, iz] != 1:
                czf = iz
                break

            for ix in range(cxi, cxf):
                if m[ix, cyi, iz] != 1:
                    cxf = ix
                    break

        for iz in range(czi, czf):
            if m[cxi, cyi + N - 1, iz] != 1:
                czf_bis = iz
                break

            for ix in range(cxi, cxf):
                if m[ix, cyi + N - 1, iz] != 1:
                    cxf_bis = ix
                    break

        cxf = min(cxf, cxf_bis)
        czf = min(czf, czf_bis)
        
        if cxf - cxi > 0 and czf - czi > 0:
            cuboids.append(dict(origin=(cxi, cyi, czi), size=(cxf - cxi, N, czf - czi)))
            for ix in range(cxi, cxf):
                for iy in range(cyi, cyi + N):
                    for iz in range(czi, czf):
                        m[ix, iy, iz] = 0
        else:
            break

    return cuboids


cdef list _get_z_3d_cuboids(integer[:, :, ::1] mask, int N):

    cdef Py_ssize_t ix, iy
    cdef short cxi, cyi, czi
    cdef short cxf, cxf_bis, cyf, cyf_bis
    cdef integer[:, :, ::1] m = mask.copy()
    cdef list cuboids = []

    while any_nonzero3d(m):

        cxi, cyi, czi = first_nonzero3d(m)
        cxf, cyf = m.shape[0], m.shape[1]
        cxf_bis, cyf_bis = m.shape[0], m.shape[1]

        for iy in range(cyi, cyf):
            if m[cxi, iy, czi] != 1:
                cyf = iy
                break

            for ix in range(cxi, cxf):
                if m[ix, iy, czi] != 1:
                    cxf = ix
                    break

        for iy in range(cyi, cyf):
            if m[cxi, iy, czi + N - 1] != 1:
                cyf_bis = iy
                break

            for ix in range(cxi, cxf):
                if m[ix, iy, czi + N - 1] != 1:
                    cxf_bis = ix
                    break

        cxf = min(cxf, cxf_bis)
        cyf = min(cyf, cyf_bis)

        if cxf - cxi > 0 and cyf - cyi > 0:
            cuboids.append(dict(origin=(cxi, cyi, czi), size=(cxf - cxi, cyf - cyi, N)))
            for ix in range(cxi, cxf):
                for iy in range(cyi, cyf):
                    for iz in range(czi, czi + N):
                        m[ix, iy, iz] = 0
        else:
            break

    return cuboids


cdef bint any_nonzero2d(integer[:, ::1] m, short val=1):
    """ Return True if (m == val).any() else False. """
    cdef bint flag = False
    cdef int nx = m.shape[0]
    cdef int ny = m.shape[1]
    cdef Py_ssize_t ix, iy

    for ix in range(nx):
        for iy in range(ny):
            if m[ix, iy] == val:
                flag = True
                break
        if flag:
            break
    return flag


cdef bint any_nonzero3d(integer[:, :, ::1] m, short val=1):
    """ Return True if (m == val).any() else False. """
    cdef bint flag = False
    cdef int nx = m.shape[0]
    cdef int ny = m.shape[1]
    cdef int nz = m.shape[2]
    cdef Py_ssize_t ix, iy, iz

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if m[ix, iy, iz] == val:
                    flag = True
                    break
            if flag:
                break
        if flag:
            break

    return flag


cdef tuple first_nonzero2d(integer[:, ::1] m, short val=1):
    """ Return the coordinate (ix, iy) of the first m == val. """
    cdef bint flag = False
    cdef int nx = m.shape[0]
    cdef int ny = m.shape[1]
    cdef Py_ssize_t ix, iy

    for ix in range(nx):
        for iy in range(ny):
            if m[ix, iy] == val:
                flag = True
                break
        if flag:
            break
    return ix, iy


cdef tuple first_nonzero3d(integer[:, :, ::1] m, short val=1):
    """ Return the coordinate (ix, iy, iz) of the first m == val. """
    cdef bint flag = False
    cdef int nx = m.shape[0]
    cdef int ny = m.shape[1]
    cdef int nz = m.shape[2]
    cdef Py_ssize_t ix, iy, iz

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if m[ix, iy, iz] == val:
                    flag = True
                    break
            if flag:
                break
        if flag:
            break

    return ix, iy, iz

cpdef int count_zero2d(integer[:, ::1] m):
    """ Count the number of 0 in m. """
    cdef int nx = m.shape[0]
    cdef int ny = m.shape[1]
    cdef int n = 0
    cdef Py_ssize_t ix, iy

    for ix in range(nx):
        for iy in range(ny):
            if m[ix, iy] == 0:
                n += 1
    return n


cpdef int count_zero3d(integer[:, :, ::1] m):
    """ Count the number of 0 in m. """
    cdef int nx = m.shape[0]
    cdef int ny = m.shape[1]
    cdef int nz = m.shape[2]
    cdef int n = 0
    cdef Py_ssize_t ix, iy, iz

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if m[ix, iy, iz] == 0:
                    n += 1
    return n