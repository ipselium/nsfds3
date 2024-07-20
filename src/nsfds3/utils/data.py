#! /usr/bin/env python3 -*- coding: utf-8 -*-
#
# Copyright © 2016-2020 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of nsfds3
#
# nsfds3 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# nsfds3 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# nsfds3. If not, see <http://www.gnu.org/licenses/>.
#
# Creation Date : 2022-07-21 - 09:00:24
"""
The module `data` provides some helper class or function to retrieve data from
nsfds3 simulations.
"""


import sys
import pathlib
import h5py
import numpy as _np
from libfds import cfdtd


VIEWS = {'e': 're', 'vx': 'ru', 'vy': 'rv', 'vz': 'rw'}


def parse_decimate(d, ndim):
    if isinstance(d, (tuple, list)):
        return d[:ndim]
    else:
        return (1, ) * ndim


def closest_index(n, ns, nt):
    """Returns the index closest to `n`.

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
    if n == 0:
        return 0

    if n > nt:
        return nt

    if n % ns == n:
        return ns

    if n % ns > ns / 2:
        return (n // ns + 1) * ns

    if n % ns <= ns / 2:
        return n // ns * ns

    return ns


def get_pressure(r=None, ru=None, rv=None, rw=None, re=None, gamma=1.4):
    """Get pressure from conservative variables.

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

    Example
    -------

    The pressure is not directly provided. To access acoustic pressure, one can
    use

    ::

        from nsfds3.utils import get_pressure

        p = get_pressure(r=r, ru=ru, rv=rv, rw=rw, re=re, gamma=gamma)

    """
    if any(item is None for item in [r, ru, rv, re]):
        raise ValueError('r, ru, rv[, rw], re must be provided')

    p = _np.empty_like(r)

    if p.ndim == 3 and rw is not None:
        fields.p_update3d(p, r, ru, rv, rw, re, gamma)
    elif p.ndim == 2:
        fields.p_update2d(p, r, ru, rv, re, gamma)
    else:
        raise ValueError('Variable must be 2 or 3d')

    return p


def get_probes(cfg, save=False):
    """Get probe location and values from cfg file.

    Parameters
    ----------
    cfg: CfgSetup
        Configuration of the simulation. Must be a CfgSetup instance.
    save: bool, default to False
        Whether to save probe to npy files or not.

    Return
    ------
    (locations, values, time): tuple
        locations are the spacial positions of the probes.
        values are the values of the probes. Each line corresponds to the values of a probe
        time is the time axis.

    Example
    -------

    ::

        from nsfds3.solver import CfgSetup
        from nsfds3.utils import get_probes

        cfg = CfgSetup('config_file.conf')
        locations, values, time = get_probes(cfg)
    """

    msh = cfg.get_grid_backup()
    if not cfg.files.data_path.is_file():
        raise FileNotFoundError('Unable to load hdf5 file.')

    with h5py.File(cfg.files.data_path, 'r') as fn:
        locs = fn['probe_locations'][...].tolist()
        if not locs:
            return _np.array([]), _np.array([]), _np.array([])
        locs = _np.array([[msh.paxis[k][tuple(c)] for k in range(cfg.ndim)] for c in locs])
        values = fn['probe_values'][...] - cfg.tp.p0
    time = _np.arange(cfg.sol.nt) * cfg.dt

    if save:
        _np.save(cfg.files.directory / f'{cfg.files.name.stem}_probes.npy', values)
        _np.save(cfg.files.directory / f'{cfg.files.name.stem}_time.npy', time)
        _np.save(cfg.files.directory / f'{cfg.files.name.stem}_locations.npy', locs)

    return locs, values, time


def check_view(view, volumic=True, vorticity=True):
    """Validates the given view parameter and returns it if valid.

    Parameters
    ----------
    view : str
        The name of the variable to consider
    volumic : bool, optional
        Specifies whether the simulation is 3d or not
    vorticity : bool, optional
        Specifies whether vorticity is calculated or not

    Returns
    -------
    str
        The validated view parameter
    """

    views = ['r', 'ru', 'rv', 'rw', 're', 'wx', 'wy', 'wz',
             'p', 'rho', 'vx', 'vy', 'vz', 'e']

    if view == 'rho':
        view = 'r'

    if not volumic and view in ['rw', 'vz', 'wx', 'wy']:
        raise ValueError('No z-component for velocity')

    elif not vorticity and view in ['wx', 'wy', 'wz']:
        raise ValueError('No data for vorticity')

    elif view not in views:
        vortis = '|wz' if volumic else '|wx|wy|wz'
        msg = 'view must be : {}' + (vortis if vorticity else '')
        var3d = 'p|r|rho|r|ru|rv|rw|re|vx|vy|vz|e'
        var2d = 'p|r|rho|r|ru|rv|re|vx|vy|e'
        raise ValueError(msg.format(var3d if volumic else var2d) )

    return view


class DataIterator:
    """Data Generator. This class is not intended for direct use.
    Use Hdf5Wrapper.get_iterator() method instead.

    Parameters
    ----------
    data: Hdf5Wrapper
        Hdf5Wrapper instance.
    view: tuple
        The variable(s) to display.
    nt: int
        Maximum number of time iterations to consider.
    step:
        Final iterator contains only 1 dataset over step.
    """

    def __init__(self, data, view=('p'), nt=None, step=1):

        if not isinstance(data, Hdf5Wrapper):
            raise ValueError('Hdf5Wrapper instance expected')

        self.data = data
        self.step = max(1, step)

        if not isinstance(view, (tuple, list)):
            view = (view, )

        self.view = view
        self.ns = self.data.get_attr('ns')
        self.icur = 0
        self.nt = self.data.get_attr('nt') if nt is None else nt
        self.nt = closest_index(self.nt, self.ns, self.nt)

    def __len__(self):
        return int((self.nt - self.icur) / (self.ns * self.step))

    def __iter__(self):
        """Iterator """
        return self

    def __next__(self):
        """Next element of iterator : (frame_number, variable) """
        if self.icur > self.nt:
            raise StopIteration

        try:
            tmp = [self.icur, ]
            for var in self.view:
                tmp.append(self.data.get(view=var, iteration=self.icur))

            self.icur += self.ns * self.step

            return tmp

        except KeyError:
            raise StopIteration


class Hdf5Wrapper:
    """Helper class to extract data from an h5py.File.

    Parameters
    ----------
    path: pathlib.Path or str
        Path to hdf5 file.

    Note
    ----
    Until the Hdf5Wrapper instance is not closed with the close method, it keeps the hdf5 file open.
    This can cause problems when opening it in another process.
    Encourage the use of the context manager :

    ::

        with Hdf5Wrapper(path_to_hdf5_file) as data:
            data.get(view='p', iteration=100)
    """

    def __init__(self, path):

        self.path = pathlib.Path(path)
        if not path.is_file():
            raise FileNotFoundError('Unable to find hdf5 file.')

        self.data = self.get_data(path)

        self.nt = self.data.attrs['nt']
        self.ns = self.data.attrs['ns']
        self.gamma = self.data.attrs['gamma']
        self.p0 = self.data.attrs['p0']
        self.volumic = self.data.attrs['ndim'] == 3
        self.vorticity = self.data.attrs['vorticity']

        self.T = (0, 1, 2) if self.volumic else (1, 0)

    def __enter__(self):
        return self

    def __exit__(self, mtype, value, traceback):
        self.close()

    def get_iterator(self, view=('p'), nt=None, step=1):
        """Return a Iterator for view(s) until nt.

        Parameters
        ----------
        view: tuple or str
            view(s) to be setup for iteration.
        nt: int
            Maximum number of time iterations to consider.
        step:
            Take only 1 dataset over step.

        Return
        ------
        DataIterator
        """
        return DataIterator(self, view=view, nt=nt, step=step)

    @staticmethod
    def get_data(fname):
        """Get data from h5py.File (hdf5 file).

        Parameters
        ----------
        fname: str, pathlib.Path
            Path to hdf5.File.

        Returns
        -------
        data: h5py.File
            The buffer interface to the hdf5 file.

        Raises
        ------
        OSError
            If the hdf5 file is not valid or does not exist.
        """

        try:
            fname = pathlib.Path(fname).expanduser()
            data = h5py.File(fname, 'r')
            return data
        except OSError:
            print('You must provide a valid hdf5 file')
            sys.exit(1)

    def reference(self, view='p', ref=None, nt=None, decimate=None, maxref=100):
        """Generate the references for min/max colormap values.

        Parameters
        ----------
        view: str
            The quantity from which the reference is to be taken
        nt: int
            Maximum number of time iterations to consider.
        ref: int, tuple, None, or str
            Can be int (frame index), tuple (int_min, int_max), or 'auto'
        decimate: tuple
            Decimation factor for each direction, default to None.
        maxref: int
            Take maxref reference linearly spaced over all datasets'.

        Returns
        -------
        tuple:
            The minimum and maximum values of the view

        Note
        ----
        Autoref can be very slow for large datasets.
        For instance for 1000 datasets of 6000x6000 array:
            - load array from hdf5 = 70ms x 1000 = 1min10sec
            - Search for min/max = 10ms x 1000 = 10sec
            - Total process = ~ 1min20sec
        Decimate by 2 the array leads to a gain of 5sec.
        Take one dataset over 2 leads to a gain of 35sec.
        """
        view = check_view(view, self.volumic, self.vorticity)

        nt = nt if nt is not None else self.nt

        if ref is None or ref == 'auto':
            ref = self._autoref(view=view, nt=nt, decimate=decimate, maxref=maxref)

        if isinstance(ref, int):
            iteration = closest_index(ref, self.ns, nt)
            var = self.get(view=view, iteration=iteration)
            return _np.nanmin(var), _np.nanmax(var)

        if isinstance(ref, tuple):
            iteration_min = closest_index(ref[0], self.ns, nt)
            iteration_max = closest_index(ref[1], self.ns, nt)
            varmin = self.get(view=view, iteration=iteration_min)
            varmax = self.get(view=view, iteration=iteration_max)
            return _np.nanmin(varmin), _np.nanmax(varmax)

    def _autoref(self, view='p', nt=None, decimate=None, maxref=100):
        """Search minimum and maximum value indices of the view.

        Parameter
        ---------
        view: str
            View from which to find reference.
        nt: int
            Maximum number of time iterations to consider.
        decimate: tuple
            Decimation factor for each direction, default to None.
        maxref: int
            Take maxref reference linearly spaced over all datasets'.

        Returns
        -------
        tuple:
            References of the minimum and maximum value for the view.
        """
        view = check_view(view, self.volumic, self.vorticity)

        # Take only maxref references.
        nt = nt if nt is not None else self.nt
        step = max(1, int(nt / (maxref * self.ns)))

        # Decimate data
        if self.volumic:
            dx, dy, dz = parse_decimate(decimate, 3)
            slices = slice(None, None, dx), slice(None, None, dy), slice(None, None, dz)
        else:
            dx, dy = parse_decimate(decimate, 2)
            slices = slice(None, None, dx), slice(None, None, dy)

        var = self.get_iterator(view=(view, ), nt=nt, step=step)
        mins, maxs = _np.array([(v[*slices].max(), v[*slices].min()) for _, v in var]).T

        refmax = abs(maxs - maxs.mean()).argmin() * self.ns * step
        refmin = abs(mins - mins.mean()).argmin() * self.ns * step

        return refmin, refmax

    def close(self):
        """Close hdf5 file."""
        self.data.close()

    def list(self):
        """List all datasets and attributes."""
        datasets = [i for i in self.data.keys() if '_it' not in i]
        print('datasets: ', *datasets)
        print('attrs: ', *list(self.data.attrs))

    def get(self, view='p', iteration=0):
        """Get data of the specified view at the given iteration."""
        iteration = closest_index(iteration, self.ns, self.nt)
        view = check_view(view, self.volumic, self.vorticity)

        if view == 'p' and self.data.get('p_it0'):
            return (self.data[f"p_it{iteration}"][...]).transpose(self.T) - self.p0

        elif view == 'p' and not self.data.get('p_it0'):
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
            v = self.data[f"{VIEWS[view]}_it{iteration}"][...]
            r = self.data[f"r_it{iteration}"][...]
            return (v / r).transpose(self.T)

        elif view in ['r', 'ru', 'rv', 'rw', 're', 'wx', 'wy', 'wz']:
            return (self.data[f"{view}_it{iteration}"][...]).transpose(self.T)

    def get_attr(self, attr):
        """Get attribute from hdf5 file. attr must be string."""
        return self.data.attrs[attr]

    def get_dataset(self, dataset):
        """Get dataset from hdf5 file. attr must be string."""
        return (self.data[dataset][...])


class FieldWrapper:
    """Helper class to extract data from a lbfds.cfdtd.cFdtd2d or Fields3d.

    Parameters
    ----------
    fld: libfds.cfdtd.cFdtd2d or libfds.cfdtd.cFdtd3d
         The field to get data from.

    Raises
    ------
        ValueError: If fld is neither Fields2d nor Fields3d.
    """



    def __init__(self, fld):
        self.fld = fld
        if isinstance(fld, cfdtd.cFdtd2d):
            self.volumic = False
            self.T = (1, 0)
        elif isinstance(fld, cfdtd.cFdtd3d):
            self.volumic = True
            self.T = (0, 1, 2)
        else:
            raise ValueError('fld must be Fields2d or Fields3d')

        self.vorticity = bool(fld.wz)

    def get(self, view='p', iteration=None):
        """
        Get the specified view.

        Parameters
        ----------
        view: str, optional
            The view to retrieve. Default is 'p'.
        iteration: int, optional
            The iteration number. Default is None.

        Returns
        -------
        numpy.ndarray: The field data corresponding to the specified view.
        """
        view = check_view(view, self.volumic, self.vorticity)

        if view == 'p':
            return _np.array(self.fld.p).transpose(self.T) - self.fld.p0

        elif view in ['r', 'ru', 'rv', 'rw', 're', 'wx', 'wy', 'wz']:
            return _np.array(getattr(self.fld, view)).transpose(self.T)

        elif view in ['vx', 'vy', 'vz', 'e']:
            return _np.array(getattr(self.fld, VIEWS[view]) / self.fld.r).transpose(self.T)