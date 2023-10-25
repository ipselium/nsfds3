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
# Creation Date : 2022-07-21 - 09:00:24
"""
The module `data` provides some helper class or function to retrieve data from nsfds3 simulations.
"""


import sys
import pathlib
import h5py
import numpy as _np
from libfds import fields


def closest_index(n, ns, nt):
    """ Returns the index closest to `n`.

    Parameters
    ----------
    n: int
        Index to look for
    ns: int
        Subdivision of `nt`
    nt: int
        Total number of iterations

    Returns
    -------
    ns: int
        The index closest to n
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


def get_data(fname):
    """ Get data from hdf5 file named `filename`.

    Parameters
    ----------
    fname: str
        path to data file

    Returns
    -------
    data: h5py file
        The buffer interface to the hdf5 file

    Raises
    ------
    OSError
        If hdf5 file is not valid or does not exist
    """

    try:
        fname = pathlib.Path(fname).expanduser()
        data = h5py.File(fname, 'r')
    except OSError:
        print('You must provide a valid hdf5 file')
        sys.exit(1)
    else:
        return data


def get_pressure(r=None, ru=None, rv=None, rw=None, re=None, gamma=1.4):
    """ Get pressure from conservative variables.

    Parameters
    ----------
    r, ru, rv, rw, re: numpy.array, optional
        Conservative variables
    gamma: float, optional
        Heat capacity ratio

    Returns
    -------
    p: numpy.array
        Pressure

    Examples
    --------

    The pressure is not directly provided. To access acoustic pressure, one can use::

        from nsfds3.utils import get_pressure

	    p = get_pressure(r=r, ru=ru, rv=rv, rw=rw, re=re, gamma=gamma)

    """
    if any(item is None for item in [r, ru, rv, re]):
        raise ValueError('r, ru, rv, re must be provided')

    p = _np.empty_like(r)

    if p.ndim == 3 and rw is not None:
        fields.update_p3d(p, r, ru, rv, rw, re, gamma)
    elif p.ndim == 2:
        fields.update_p2d(p, r, ru, rv, re, gamma)
    else:
        raise ValueError('Variable must be 2 or 3d')

    return p


def check_view(view, volumic=True, vorticity=True):
    """ Displays a message specifying which variables can be plotted.

    Parameters
    ----------
    view: str
        The name of the variable to consider
    volumic: bool
        Specifies whether the simulation is 3d or not
    vorticity:
        Specifies whether vorticity is calculated or not
    """

    views = ['r', 'ru', 'rv', 'rw', 're', 'wx', 'wy', 'wz',
             'p', 'rho', 'vx', 'vy', 'vz', 'e']

    if view == 'rho':
        view = 'r'

    if view in ['rw', 'vz', 'wx', 'wy'] and not volumic:
        print('No z-component for velocity')
        sys.exit(1)

    if view in ['wx', 'wy', 'wz'] and not vorticity:
        print('No data for vorticity')
        sys.exit(1)

    if view not in views:
        vortis = '|wz' if volumic else '|wx|wy|wz'
        msg = 'view must be : {}' + (vortis if vorticity else '')
        var3d = 'p|r|rho|r|ru|rv|rw|re|vx|vy|vz|e'
        var2d = 'p|r|rho|r|ru|rv|re|vx|vy|e'
        print(msg.format(var3d if volumic else var2d) )
        sys.exit(1)

    return view


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

        if not isinstance(view, (tuple, list)):
            view = (view, )

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

        self.var = {'e': 're', 'vx': 'ru', 'vy': 'rv', 'vz': 'rw'}
        self.nt = self.get_attr('nt')
        self.ns = self.get_attr('ns')

        self.gamma = self.get_attr('gamma')
        self.p0 = self.data.attrs['p0']

        self.volumic = self.get_attr('ndim') == 3
        self.vorticity = self.get_attr('vorticity')

        if self.volumic:
            self.T = (0, 1, 2)
        else:
            self.T = (1, 0)

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

        view = check_view(view, self.volumic, self.vorticity)

        if not ref or ref == 'auto':
            ref = self.autoref(view=view)

        if isinstance(ref, int):
            var = self.get(view=view,
                           iteration=closest_index(ref, self.ns, self.nt))
            return _np.nanmin(var), _np.nanmax(var)

        if isinstance(ref, tuple):
            varmin = self.get(view=view,
                              iteration=closest_index(ref[0], self.ns, self.nt))
            varmax = self.get(view=view,
                              iteration=closest_index(ref[1], self.ns, self.nt))
            return _np.nanmin(varmin), _np.nanmax(varmax)

    def autoref(self, view='p'):
        """ Autoset reference. """

        view = check_view(view, self.volumic, self.vorticity)
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

        view = check_view(view, self.volumic, self.vorticity)

        if view == 'p':
            r = self.data[f"r_it{iteration}"][...]
            ru = self.data[f"ru_it{iteration}"][...]
            rv = self.data[f"rv_it{iteration}"][...]
            re = self.data[f"re_it{iteration}"][...]
            if self.volumic:
                rw = self.data[f"rw_it{iteration}"][...]
            else:
                rw = None
            p = get_pressure(r=r, ru=ru, rv=rv, rw=rw, re=re, gamma=self.gamma)
            return p.transpose(self.T) - self.p0

        elif view in ['vx', 'vy', 'vz', 'e']:
            v = self.data[f"{self.var[view]}_it{iteration}"][...]
            r = self.data[f"r_it{iteration}"][...]
            return (v / r).transpose(self.T)

        elif view in ['r', 'ru', 'rv', 'rw', 're', 'wx', 'wy', 'wz']:
            return (self.data[f"{view}_it{iteration}"][...]).transpose(self.T)

    def get_attr(self, attr):
        """ Get attribute from hdf5 file. attr must be string."""
        return self.data.attrs[attr]

    def get_dataset(self, dataset):
        """ Get dataset from hdf5 file. attr must be string."""
        return self.data[dataset][...]


class FieldExtractor:

    def __init__(self, fld):

        self.fld = fld

        if isinstance(fld, fields.Fields2d):
            self.volumic = False
            self.T = (1, 0)
        elif isinstance(fld, fields.Fields3d):
            self.volumic = True
            self.T = (0, 1, 2)
        else:
            raise ValueError('fld must be Fields2d or Fields3d')


        self.vorticity = False if len(fld.wz) == 0 else True

        self.var = {'e': 're', 'vx': 'ru', 'vy': 'rv', 'vz': 'rw'}

    def get(self, view='p', iteration=None):
        """ Get data at iteration. """

        view = check_view(view, self.volumic, self.vorticity)

        if view == 'p':
            return _np.array(self.fld.p).transpose(self.T) - self.fld.p0

        if view in ['r', 'ru', 'rv', 'rw', 're', 'wx', 'wy', 'wz']:
            return _np.array(getattr(self.fld, view)).transpose(self.T)

        if view in ['vx', 'vy', 'vz', 'e']:
            return _np.array(getattr(self.fld, self.var[view]) / self.fld.r).transpose(self.T)