#! /usr/bin/env python3
# -*- coding: utf-8 -*-
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
# Creation Date : 2019-03-21 - 23:43:11
"""
Some utils to

    * get attributes from .py files
    * get configuration and mesh objects from pickles
    * read/write files or create directories
"""

import os
import sys
import pickle
import pathlib
import time
import scipy.io.wavfile as _wf
import numpy as _np
from nsfds3.utils.data import DataExtractor as _DataExtractor
from nsfds3.cpgrid import templates as _tplt


def get_func(module, name):
    """ Get obstacle from custom file or fdgrid templates. """

    if os.path.isfile(module):
        sys.path.append(os.path.dirname(module))
        custom = __import__(os.path.basename(module).split('.')[0])
    else:
        custom = _tplt.TestCases

    try:
        return getattr(custom, name)
    except AttributeError:
        return None


def get_objects(path, fname):
    """ Get cfg and msh objects from pickles.

    Parameters
    ----------
    path: str
        path to cfg and mesh files
    fname: str
        basename for cfg and msh files

    Returns
    -------
    tuple of cfg and msh: tuple
        Returns the cfg and msh objects

    Examples
    --------
    `.cfg` and `.msh` files are created for each simulation.
    They contain the configuration and grid objects relative to the simulation.
    To retrieve them, simply do::

        from nsfds3.utils import get_objects

        cfg, msh = get_objects('path_to_cfg_and_msh_files', 'basename_of_these_files')

    """
    path = pathlib.Path(path)
    fname = pathlib.Path(fname)

    with open(path / fname.with_suffix('.cfg'), 'rb') as f:
        cfg = pickle.load(f, encoding='bytes')

    with open(path / fname.with_suffix('.msh'), 'rb') as f:
        msh = pickle.load(f, encoding='bytes')

    return cfg, msh


def mkdir(directory, verbose=False):
    """ Check if directory exists. If not, create it.

    Parameters
    ----------
        directory: pathlib.Path or str
            Directory to create
        verbose: bool
            Verbose mode
    """
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)

    if not directory.is_dir():
        directory.mkdir()
        if verbose:
            print(f"Create directory : {directory}")
            time.sleep(0.5)
    elif verbose:
        print(f"{directory} already exits")
        time.sleep(0.5)



def write_file(fname, content, verbose=False):
    """ Write content into fname.

    Parameters
    ----------
    fname: pathlib.Path or str
        Path to fname
    content: str
        Content to write to fname
    verbose: bool
        Verbose mode
    """
    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)

    if fname.parent.is_dir():
        with open(fname, 'w', encoding="utf-8") as f:
            f.write(content)
        if verbose:
            print(f'{fname.name} created')
            time.sleep(0.5)
    elif verbose:
        print(f'{fname.parent} does not exist')
        time.sleep(0.5)


def read_file(fname, verbose=False):
    """ Return the content of fname.

    Parameters
    ----------
    fname: pathlib.Path or str
        Path to fname
    verbose: bool
        Verbose mode

    Returns
    -------
    content: str
        Content of fname
    """
    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)

    if fname.is_file():
        with open(fname, 'r', encoding="utf-8") as f:
            content = f.read()
        if verbose:
            print(f'{fname.name} is read')
            time.sleep(0.5)
        return content
    elif verbose:
        print(f'{fname.name} does not exist')
        time.sleep(0.5)
    return ''


def normalize(s, dtype=_np.int16):
    """ Normalize wav data. """
    smax = _np.iinfo(dtype).max
    return (s/abs(s).max()*smax).astype(dtype)


def probes_to_wavfile(datafile, dtype=_np.int16, path=None):
    """ Make .wav files from probes signals.

    Parameters
    ----------
    datafile: hdf5 file
        the hdf5 file
    dtype:
        type of the output wave file (np.int8, np.int16)
    path:
        path to save wavefiles

    Returns
    -------
    out: list
        list of probe signals saved in the wave files
    """

    probes = []
    datafile = pathlib.Path(datafile).expanduser()

    with _DataExtractor(datafile) as data:
        probe_values = data.get_dataset('probe_values')
        p0 = data.get_attr('p0')
        dt = data.get_attr('dt')

    if not path:
        path = os.path.dirname(datafile)


    if list(probe_values):

        filename = os.path.basename(datafile).split('.')[0]

        for i, p in enumerate(probe_values):
            name = f'{path}/{filename}_probe_{i}.wav'
            tmp = normalize(p - p0, dtype=dtype)
            print(f'Writing {name}...')
            _wf.write(name, rate=int(1/dt), data=tmp)
            probes.append(tmp)
        print('Done!')

    else:
        print('No probes found!')

    return probes