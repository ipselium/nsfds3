#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2020 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of {name}
#
# {name} is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# {name} is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with {name}. If not, see <http://www.gnu.org/licenses/>.
#
# Creation Date : 2022-07-21 - 09:00:24
"""
-----------
DOCSTRING

-----------
"""


import sys
import pathlib
import h5py
import numpy as _np
from libfds import fields as _fld


def nearest_index(n, ns, nt):
    """ Returns nearest possible index `n`

    Parameters
    ----------
    n: int
        Index to look for
    ns: int
        Subdivision of `nt`
    nt: int
        Total number of iterations
    """

    if n > nt:
        return nt

    if n % ns == n:
        return ns

    if n % ns > ns / 2:
        return (n // ns + 1) * ns

    if n % ns <= ns / 2:
        return n // ns * ns

    return ns


def get_data(filename):
    """ Get data from `filename` (hdf5 file). """

    try:
        filename = pathlib.Path(filename).expanduser()
        data = h5py.File(filename, 'r')
    except OSError:
        print('You must provide a valid hdf5 file')
        sys.exit(1)
    else:
        return data


class DataIterator:
    """ Data Generator

    Parameters
    ----------
    data : DataExtractor or str
        DataExtractor instance or filename.
    view : tuple
        The variable(s) to display.
    nt : int
        The last frame to consider.

    """

    def __init__(self, data, view=('p'), nt=None):

        if isinstance(data, DataExtractor):
            self.data = data
        elif isinstance(data, (pathlib.Path, str)):
            self.data = DataExtractor(data)

        self.view = view
        self.ns = self.data.get_attr('ns')
        self.icur = 0
        if nt is None:
            self.nt = self.data.get_attr('nt')
        else:
            self.nt = nt

    def __len__(self):
        return int((self.nt - self.icur) / self.ns)

    def __iter__(self):
        """ Iterator """
        return self

    def __next__(self):
        """ Next element of iterator : (frame_number, variable) """
        try:
            if self.icur > self.nt:
                raise StopIteration

            tmp = [self.icur]
            for var in self.view:
                tmp.append(self.data.get(view=var, iteration=self.icur))

            self.icur += self.ns

            return tmp

        except KeyError:
            raise StopIteration


class DataExtractor:
    """ Extract data from hdf5 file

    Parameters
    ----------
    data : str, hdf5file
        Path to hdf5 file or data from hdf5 file.

    """

    def __init__(self, data):

        if isinstance(data, (pathlib.Path, str)):
            self.data = get_data(data)
        else:
            self.data = data

        self.var = {'p': 'p', 'rho': 'r', 'vx': 'ru', 'vy': 'rv', 'e': 're', 'vxyz': 'vxyz'}
        self.nt = self.get_attr('nt')
        self.ns = self.get_attr('ns')

        if self.get_attr('mesh') == 'curvilinear':
            self.J = self.get_dataset('J')
        else:
            self.J = _np.ones((self.get_attr('nx'), self.get_attr('ny')))

    def __enter__(self):
        return self

    def __exit__(self, mtype, value, traceback):
        self.close()

    def reference(self, view='p', ref=None):
        """ Generate the references for min/max colormap values

        Parameters
        ----------
        view : str
            The quantity from which the reference is to be taken
        ref : int, tuple, None, or str
            Can be int (frame index), tuple (int_min, int_max), or 'auto'
        """

        if not ref or ref == 'auto':
            ref = self.autoref(view=view)

        if isinstance(ref, int):
            var = self.get(view=view,
                           iteration=nearest_index(ref, self.ns, self.nt))
            return _np.nanmin(var), _np.nanmax(var.max)

        if isinstance(ref, tuple):
            varmin = self.get(view=view,
                              iteration=nearest_index(ref[0], self.ns, self.nt))
            varmax = self.get(view=view,
                              iteration=nearest_index(ref[1], self.ns, self.nt))
            return _np.nanmin(varmin), _np.nanmax(varmax)

        print("Only 'p', 'rho', 'vx', 'vy', 'vxyz' and 'e' available !")
        sys.exit(1)

    def autoref(self, view='p'):
        """ Autoset reference. """

        var = DataIterator(self, view=(view, ))

        maxs = []
        mins = []
        for _, v in var:
            maxs.append(v.max())
            mins.append(v.min())

        maxs = _np.array(maxs)
        mins = _np.array(mins)

        refmax = abs(maxs - maxs.mean()).argmin() * self.ns
        refmin = abs(mins - mins.mean()).argmin() * self.ns

        return refmin, refmax

    def close(self):
        """ Close hdf5 file """
        self.data.close()

    def list(self):
        """ List all datasets and attributes. """

        datasets = [i for i in self.data.keys() if '_it' not in i]
        print('datasets: ', *datasets)
        print('attrs: ', *list(self.data.attrs))

    def get(self, view='p', iteration=0):
        """ Get data at iteration. """

        if view == 'p':
            r = self.data[f"r_it{iteration}"][...] * self.J
            ru = self.data[f"ru_it{iteration}"][...] * self.J
            rv = self.data[f"rv_it{iteration}"][...] * self.J
            re = self.data[f"re_it{iteration}"][...] * self.J
            p = _np.empty_like(r)
            _fld.update_p(p, r, ru, rv, re, self.data.attrs['gamma'])

            return p.T - self.data.attrs['p0']

        if view in ['rho', 'vxyz']:
            try:
                return (self.data[f"{self.var[view]}_it{iteration}"][...] * self.J).T
            except KeyError:
                print('No data for vorticity')
                sys.exit(1)

        if view in ['vx', 'vy', 'e']:
            vx = (self.data[f"{self.var[view]}_it{iteration}"][...] * self.J)
            r = (self.data[f"r_it{iteration}"][...] * self.J)
            return (vx / r).T

        raise ValueError("view must be 'p', 'rho', 'vx', 'vy', 'e', or 'vxyz'")

    def get_attr(self, attr):
        """ Get attribute from hdf5 file. attr must be string."""
        return self.data.attrs[attr]

    def get_dataset(self, dataset):
        """ Get dataset from hdf5 file. attr must be string."""

        return self.data[dataset][...]
