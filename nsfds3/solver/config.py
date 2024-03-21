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
#
# Creation Date : 2016-11-29 - 23:18:27
#
# pylint: disable=too-many-statements
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-instance-attributes
"""
The `config` module contains the :py:class:`CfgSetup` that read the configuration file
and set all simulation parameters.

Example
-------

::

    from nsfds3.init import CfgSetup

    cfg_a = CfgSetup()
    cfg_b = CfgSetup(path_to_configuration_file)
"""

import os
import ast
import sys
import pathlib
import numpy
from rich import print
from configparser import ConfigParser
from pkg_resources import parse_version as _parse_version
from nsfds3.utils import files, misc
from nsfds3.solver import sources
from nsfds3.materials import Air
from nsfds3.cpgrid import utils as cputils
from matplotlib.ticker import EngFormatter


class Files:

    def __init__(self, path, directory, name, overwrite=False, verbose=False):

        self.path = path
        self._directory = directory
        self._name = name
        self.overwrite = overwrite
        self.verbose = verbose
        files.mkdir(self.directory, self.verbose)

    @property
    def data_path(self):
        """Absolute path to data file.

        Note
        ----
        data_path is read only. It cannot be set directly. Instead, set `directory` and `name`.
        """
        return self.directory / self.name

    @property
    def name(self):
        """Data filename used to save fields."""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, (str, pathlib.Path)):
            raise ValueError('files.name: str or pathlib.Path expected')
        self._name = pathlib.Path(value).with_suffix('.hdf5')

    @property
    def directory(self):
        """Directory where to save data files.

        Note
        ----
        If directory does not exist, create it.
        """
        return self._directory

    @directory.setter
    def directory(self, value):
        if not isinstance(value, (str, pathlib.Path)):
            raise ValueError('directory: str or pathlib.Path expected')
        if isinstance(value, str):
            value = pathlib.Path(value)
        self._directory = self.path / value
        files.mkdir(self.directory, self.verbose)

    def __str__(self):
        return 'Files'

    def __repr__(self):
        return str(self)


class Graphics:
    """Helper class used by CfgSetup to setup graphical options."""

    def __init__(self, fig=True, prb=True, bz=True, bc=True, fps=24):

        self.fig = fig
        self.prb = prb
        self.bz = bz
        self.bc = bc
        self.fps = fps


class Solver:
    """Helper class used by CfgSetup to setup solver options."""

    def __init__(self, vsc=True, cpt=True, vrt=True, flt=True,
                 xnu_n=0.2, xnu_0=0.01, nt=50, ns=10, cfl=0.5,
                 save=True, mp=False, nan_check=False, resume=False, timings=False):

        self.vsc = vsc
        self.cpt = cpt
        self.vrt = vrt
        self.flt = flt
        self._xnu_n = xnu_n
        self._xnu_0 = xnu_0
        self._nt = nt
        self._ns = ns
        self.cfl = cfl
        self.save = save
        self.it = 0
        self.itmax = 0
        self.mp = mp
        self.nan_check = nan_check
        self.resume = resume
        self.timings = timings
        self._check_filter(self.xnu_n)
        self._check_filter(self.xnu_0)

    def _check_filter(self, strength):
        """Check that the strength of the filter is set correctly."""
        if not isinstance(strength, float):
            raise ValueError('Solver.xnu: float expected')

        if not 0 <= strength <= 1:
            raise ValueError('Solver.xnu: strength must be between 0 and 1')

    @property
    def nt(self):
        """Number of time iterations.

        Note
        ----
        The value of `nt` declared here can be different from the value of `nt`
        taken into account for the simulation so that the backup frequency `ns`
        is a multiple of `nt`
        """
        return self._nt

    @nt.setter
    def nt(self, value):
        if not isinstance(value, int):
            raise ValueError('nt: integer expected')
        self._nt = value
        self.adjust_nt()

    @property
    def ns(self):
        """Field backup frequency.

        Note
        ----
        If `ns` is modified, `nt` is automatically updated to be a multiple of `ns`.
        """
        return self._ns

    @ns.setter
    def ns(self, value):
        if not isinstance(value, int):
            raise ValueError('ns: integer expected')
        self._ns = value
        self.adjust_nt()

    @property
    def xnu_n(self):
        """Selective filter strength.

        Note
        ----
        Must be between 0 and 1
        """
        return self._xnu_n

    @xnu_n.setter
    def xnu_n(self, value):
        self._check_filter(value)
        self._xnu_n = value

    @property
    def xnu_0(self):
        """Selective filter strength for points close to boundaries.

        Note
        ----
        Must be between 0 and 1
        """
        return self._xnu_0

    @xnu_0.setter
    def xnu_0(self, value):
        self._check_filter(value)
        self._xnu_0 = value

    def adjust_nt(self):
        """Adjust the number of time iterations `nt` to be a multiple of `ns`."""
        if self._nt % self._ns:
            self._nt -= self._nt % self._ns

    def __eq__(self, other):
        """Report whether self and other are the same."""
        return misc.are_equals(self, other, attrs=['ns', 'cfl'])

    def __str__(self):
        s = "\n[Solver]"
        s += f"\n\t- Viscous fluxes   : {self.vsc}"
        s += f"\n\t- Selective filter : {self.flt} [nu_0={self.xnu_0}, nu_n={self.xnu_n}]"
        s += f"\n\t- Shock capture    : {self.cpt}"
        s += f"\n\t- Vorticity        : {self.vrt}"
        return s

    def __repr__(self):
        return str(self)


class Geometry:
    """Helper class used by CfgSetup to setup geometrical parameters."""

    def __init__(self, shape, steps=None, origin=None, bc=None, flat=None,
                 path=None, file=None, name=None, kwargs=None, curvname=None,
                 bz_n=20, bz_stretch_factor=2, bz_stretch_order=3, bz_filter_order=3.,
                 stencil=11, force=False, free=True):

        self.stencil = stencil
        self.free = free
        self.force = force
        self.bz_n = bz_n
        self.bz_filter_order = bz_filter_order
        self.bz_stretch_order = bz_stretch_order
        self.bz_stretch_factor = bz_stretch_factor
        self._parse_geo(shape, steps, origin, bc, bz_n)

        self._flat = flat
        self._check_flat()

        self.path = pathlib.Path(path) if path else pathlib.Path()
        self._file = file
        self._name = name
        self._kwargs = kwargs if isinstance(kwargs, dict) else dict()
        self._curvname = curvname

        self._obstacles = None
        self._update_obstacles()
        self._update_curvilinear_transformation()

    @property
    def shape(self):
        """Shape of the computation domain."""
        if self.flat:
            ax, _ = self.flat
            return self._to_2d_tuple(self._shape, ax)
        return self._shape

    @shape.setter
    def shape(self, value):
        value = self._parse_tuple(value, 'shape')
        self._parse_geo(value, self._steps, self._origin, self._bc, self.bz_n)

    @property
    def steps(self):
        """Spatial steps (dx, dy[, dz]). Must be >= 1e6 m.

        Note
        ----
        For simulation with thermoviscous effects, a small spatial step can lead
        to instabilities.

        Note
        ----
        If `steps` is modified, the time step `dt` is modified too.
        """
        if self.flat:
            ax, _ = self.flat
            return self._to_2d_tuple(self._steps, ax)
        return self._steps

    @steps.setter
    def steps(self, value):
        value = self._parse_tuple(value, 'steps')
        self._parse_geo(self._shape, value, self._origin, self._bc, self.bz_n)

    @property
    def origin(self):
        """Origin of the computation domain."""
        if self.flat:
            ax, _ = self.flat
            return self._to_2d_tuple(self._origin, ax)
        return self._origin

    @origin.setter
    def origin(self, value):
        value = self._parse_tuple(value, 'origin')
        self._parse_geo(self._shape, self._steps, value, self._bc, self.bz_n)

    @property
    def bc(self):
        """Boundary condition of the computation domain."""
        if self.flat:
            ax, _ = self.flat
            return ''.join(bc for i, bc in enumerate(self._bc) if i not in [2*ax, 2*ax + 1])
        return self._bc

    @bc.setter
    def bc(self, value):
        if not isinstance(value, str):
            raise ValueError(f'Geometry.shape: str expected')
        if self.flat and len(value) == 4:
            ax, _ = self.flat
            value = value[:2*ax] + self._bc[2*ax:2*ax+2] + value[2*ax:]
        if len(value) != len(self._bc):
            raise ValueError(f'Geometry.bc must be of length {len(self._bc)}')
        self._parse_geo(self._shape, self._steps, self._origin, value, self.bz_n)

    @property
    def obstacles(self):
        """Obstacles in presence."""
        if self.flat and self._obstacles:
            ax, idx = self.flat
            return [obs.flatten(ax) for obs in self._obstacles if idx in obs.rn[ax]]
        return self._obstacles

    @property
    def flat(self):
        """Describe how a 3d configuration parameters are converted to a 2d configuration.

        flat: tuple (ax, idx)
            ax corresponds to the dimension to be removed, and idx to the index following
            this dimension where to take the cross-section.

        Note
        ----
        flat attribute is read only. It is not possible to change the way a 3d configuration
        is converted into a 2d configuration, or even to switch back from a 2d configuration
        to a 3d configuration.
        """
        return self._flat

    @property
    def file(self):
        """Name of the file in which to search for functions `name`,
        `curvname`, and functions used for sources.
        """
        return self._file

    @file.setter
    def file(self, value):
        self._file = value
        self._update_obstacles()
        self._update_curvilinear_transformation()

    @property
    def name(self):
        """Name of the function to be used to set up the `Obstacle` arrangement."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._update_obstacles()

    @property
    def kwargs(self):
        """Keyword arguments of the function to be used to set up the `Obstacle` arrangement."""
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        if not isinstance(value, dict):
            raise ValueError('Geometry.kwargs: dict expected')
        self._kwargs = value
        self._update_obstacles()

    @property
    def curvname(self):
        """Name of the function to be used to set up the curvilinear transformation."""
        return self._curvname

    @curvname.setter
    def curvname(self, value):
        self._curvname = value
        self._update_curvilinear_transformation()

    @property
    def args(self):
        """Return (arg, kwargs) needed to instantiate `CartesianGrid` or `CurvilinearGrid`."""
        args = self.shape, self.steps
        kwargs = {'origin': self.origin,
                  'bc': self.bc,
                  'obstacles': self.obstacles,
                  'bz_n': self.bz_n,
                  'bz_stretch_factor': self.bz_stretch_factor,
                  'bz_stretch_order': self.bz_stretch_order,
                  'bz_filter_order': self.bz_filter_order,
                  'stencil': self.stencil,
                  'free': self.free}
        if self.curvfunc:
            kwargs['curvfunc'] = self.curvfunc
        return args, kwargs

    def _parse_geo(self, shape, steps, origin, bc, bz_n):
        self._shape, self._steps, self._origin, self._bc = cputils.parse_geo(shape, steps, origin, bc, bz_n)

    def _parse_tuple(self,value, name):
        var = getattr(self, f'_{name}')
        if not isinstance(value, (tuple, list)):
            raise ValueError(f'Geometry.{name}: tuple expected')
        if self.flat and len(value) == 2:
            ax, _ = self.flat
            value = value[:ax] + (var[ax], ) + value[ax:]
        if len(value) != len(var):
            raise ValueError(f'Geometry.{name} must be of length {len(var)}')
        return value

    @staticmethod
    def _to_2d_tuple(var, ax):
        """Return a 2d version of the tuple `var` removing the ax-th value."""
        return tuple(s for i, s in enumerate(var) if i != ax)

    def _check_flat(self):
        """Check that flat is consistent."""
        if not isinstance(self._flat, (tuple, type(None))):
            raise ValueError('Geometry.flat: tuple of None expected')

        if self._flat is not None:

            if len(self._flat) != 2:
                raise ValueError('Geometry.flat: length 2 expected (axis, location)')

            if len(self._shape) != 3:
                raise ValueError('Geometry.shape: length 3 expected')

            flat_ax, flat_idx = self.flat

            if flat_ax not in range(3):
                raise ValueError('Geometry.flat[0]: 0, 1, or 2 axis expected')

            if flat_idx not in range(self._shape[flat_ax]):
                raise ValueError('Geometry.flat[1]: index must be in the domain')

    def _update_obstacles(self):
        if self.name is not None:
            file = '' if self.file is None else self.file
            self._obstacles = cputils.get_func(self.path / file, self.name)
            if self._obstacles is not None:
                self._obstacles = self._obstacles(self._shape, **self.kwargs)

        self._obstacles = [] if self._obstacles is None else self._obstacles

    def _update_curvilinear_transformation(self):
        if self.curvname is not None:
            file = '' if self.file is None else self.file
            self.curvfunc = cputils.get_func(self.path / file, self.curvname)
        else:
            self.curvfunc = None

    def __eq__(self, other):
        """Report whether self and other have same grid configuration or not."""

        attrs = ['shape', 'steps', 'origin', 'bc', 'obstacles',
                 'bz_n', "bz_stretch_factor", "bz_stretch_order", 'bz_filter_order',
                 "stencil", "curvfunc"]

        return misc.are_equals(self, other, attrs)

    def __str__(self):

        if self.obstacles:
            wall_source = any('V' in o.bc for o in self.obstacles)
        else:
            wall_source = False

        s = f"\n[{'CartesianGrid' if not self.curvfunc else 'CurvilinearGrid'}]"
        s += f"\n\t- Grid                : {'x'.join(str(n) for n in self.shape)} points grid"
        s += f'\n\t- boundary conditions : {self.bc}'
        s += f"\n\t- Spatial step        : ({', '.join(str(n) for n in self.steps)})"
        s += f"\n\t- Origin              : ({', '.join(str(n) for n in self.origin)})"
        if wall_source:
            s += f"\n\t- Wall source         : ({wall_source})"
        if 'A' in self.bc:
            s += f'\n\t- Buffer zone         : {self.bz_n} grid points'
        if self.obstacles:
            s += f"\n\t- Obstacles           : {self.obstacles}"
        if hasattr(self, 'curvfunc'):
            s += f"\n\t- Curvilinear         : {self.curvfunc}"

        return s

    def __repr__(self):
        return str(self)


class Probes:
    """Helper class used by CfgSetup to setup probes parameters.

    Parameters
    ----------
    vars: tuple
        Variables to get value from. Can be p, vx, vy[, vz], rho.
    locs: tuple
        Locations of the probes. Must be a sequence containing length n tuples
        with n being the number of dimensions of the domain (2d/3d).
    shape: tuple
        Shape of the domain.
    flat: tuple or None
        Whether geometry is flattened or not.
        If a tuple is provided, it must provide (flat axis, flat index).

    Example
    -------

    ::

        # p-probe @ (34, 49) and vx-probe @ (44, 59)
        prb = Probes(vars=(p, vx), locs=((34, 49), (44, 59)), shape=(200, 200))
    """
    def __init__(self, vars, locs, shape, flat=None):

        self._shape = shape
        self._ndim = len(shape)
        self._flat = flat
        if self._ndim != 3 and self._flat:
            raise ValueError("Source: length 3 expected for shape")

        self._vars = vars
        self._locs = self._parse_locs(locs)

    @property
    def vars(self):
        """Variables to probe."""
        return self._vars

    @vars.setter
    def vars(self, value):
        self._vars = value

    @property
    def locs(self):
        """Locations of the probes."""
        if self._flat:
            ax, _ = self._flat
            return tuple(tuple(c for i, c in enumerate(loc) if i != ax) for loc in self._locs)
        return self._locs

    @locs.setter
    def locs(self, value):
        value = self._parse_locs(value)
        if self._flat and any(len(c) == 2 for c in value):
            ax, idx = self._flat
            value = tuple(c[:ax] + (idx, ) + c[ax:] for c in value)
        self._locs = value

    def _parse_locs(self, value):
        """Parse locations of the probes."""
        # is tuple ?
        if not isinstance(value, (tuple, list)):
            raise ValueError('Probes.locs: tuple expected')

        # is tuple of tuples
        if not any(isinstance(p, (tuple, list)) for p in value):
            value = value,
        value = tuple(tuple(c) for c in value)

        # is length ok and location in the domain ?
        if hasattr(self, '_locs'):
            ndims = (0, 2, 3) if self._flat else (0, self._ndim)
        else:
            ndims = (0, self._ndim)
        for i, loc in enumerate(value):
            if len(loc) not in ndims:
                raise ValueError(f"Probes.locs: tuples of length {'|'.join(str(c) for c in ndims)} expected")
            if any(not 0 <= c < s for c, s in zip(loc, self._shape)):
                raise ValueError(f'Probes.locs[{i}]: out of bounds')

        return value

    def __len__(self):
        if not any(self.locs):
            return 0
        return len(self.locs)

    def __eq__(self, other):
        """Report whether self and other are the same."""
        return misc.are_equals(self, other, attrs=['locs', 'vars'])

    def __iter__(self):
        return iter(self.locs)

    def __str__(self):
        return f'\n[Probes]\n\t- variables : {self.vars}\n\t- Locations: {self.locs}'

    def __repr__(self):
        return str(self)


class CfgSetup:
    """Setup configuration of the solver.

    Parameters
    ----------
    cfgfile: str, optional
        If None, CfgSetup takes defaults values.
        If a valid configuration file is provided, CfgSetup takes the value contained in this file.
    last: bool, optional
        If cfgfile is not provided, try to load the last configuration used if it exists.
    verbose: bool, optional
        Verbose mode

    Note
    ----
    Hereafter, one can find the correspondance between entries in the configuration file (on the left)
    and `CfgSetup` attributes (on the right) and their default values.

    ::

        [general]
        version                     -> self.version
        quiet                       -> self.quiet = False

        [files]
        directory                   -> self.files.directory = 'data/'
        name                        -> self.files.name = 'tmp'
        overwrite                   -> self.files.overwrite = False

        [thermophysics]
        norm                        -> self.tp.norm = False
        rho0                        -> self.tp.rho0 = 101325.0
        t0                          -> self.tp.T0 - self.tp.Tref = 20.0
        gamma                       -> self.tp.gamma = 1.4

        [geometry]
        force                       -> self.geo.force = False
        free                        -> self.geo.free = True
        file                        -> self.geo.file = None
        name                        -> self.geo.name = None
        kwargs                      -> self.geo.kwargs = None
        curvname                    -> self.geo.curvname = None
        bc                          -> self.geo.bc = 'WWWWWW'
        shape                       -> self.geo.shape = (128, 96, 32)
        origin                      -> self.geo.origin = None
        steps                       -> self.geo.steps = (1., 1., 1.)
        flat                        -> self.geo.flat = None
        bz grid points              -> self.geo.bz_n = 20
        bz filter order             -> self.geo.bz_filter_order = 3
        bz stretch order            -> self.geo.bz_stretch_order = 3
        bz stretch factor           -> self.geo.bz_stretch_factor = 2

        [sources]
        on                          -> self.src.on = False
        origin                      -> self.src.origin = (),
        S0                          -> self.src.S0 = ()
        Bx                          -> self.src.Bx = ()
        By                          -> self.src.By = ()
        Bz                          -> self.src.Bz = ()
        kx                          -> self.src.kx = ()
        ky                          -> self.src.ky = ()
        kz                          -> self.src.kz = ()
        k                           -> self.src.k = ()
        R                           -> self.src.Rx = ()
        evolution                   -> self.src.evolution = ()

        [flow]
        type                        -> self.flw.ftype = None
        components                  -> self.flw.components = (0, 0, 0)

        [probes]
        variables                   -> self.prb.vars = ()
        locations                   -> self.prb.locs = ()

        [solver]
        mp                          -> self.sol.mp = False
        timings                     -> self.sol.timings = False
        nan_check                   -> self.sol.nan_check = False
        resume                      -> self.sol.resume = False
        viscous fluxes              -> self.sol.vsc = True
        vorticity                   -> self.sol.vrt = True
        shock capture               -> self.sol.cpt = True
        selective filter            -> self.sol.flt = True
        selective filter n-strength -> self.sol.xnu_n = 0.2
        selective filter 0-strength -> self.sol.xnu_0 = 0.01
        nt                          -> self.sol.nt = 50
        ns                          -> self.sol.ns = 10
        cfl                         -> self.sol.cfl = 0.5
        save fields                 -> self.sol.save = True

        [graphics]
        figures                     -> self.gra.fig = True
        probes                      -> self.gra.prb = True
        bz                          -> self.gra.bz = True
        bc                          -> self.gra.bc = True
        fps                         -> self.gra.fps = 24
    """

    _NONE = ('', 'no', 'No', 'none', 'None', None, 'False', 'false', False)
    _SECTIONS = ('general', 'files', 'thermophysic', 'geometry',
                 'sources', 'flow', 'probes', 'solver', 'graphics')

    def __init__(self, cfgfile=None, last=False, verbose=False):

        # Global attributes
        self._version_base = '0.1.0'
        self.stencil = 11
        self.verbose = verbose
        self.cpu_count = os.cpu_count()

        # Path to config files
        self.path_nsfds3 = pathlib.Path.home() / '.nsfds3'
        files.mkdir(self.path_nsfds3, self.verbose)
        self.path_last = self.path_nsfds3 / 'last'
        self.path_current = pathlib.Path.cwd()
        self.cfgfile_last = pathlib.Path(files.read_file(self.path_last, self.verbose))

        # Initialize configparser
        self._cfg = ConfigParser(allow_no_value=True, converters={'lit': self.parse_literal})

        # load the configuration file and parse all parameters
        self.load(cfgfile, last)

    def load(self, cfgfile=None, last=False):
        """Load configuration file `cfgfile`. If file is not found, fallback to default configuration.

        Parameters
        ----------
        cfgfile: str, optional
            If None, CfgSetup takes defaults values.
            If a valid configuration file is provided, CfgSetup takes the value contained in this file.
        last: bool, optional
            If cfgfile is not provided or not found, try to load the last configuration used if it exists and if last is True.
        """
        if cfgfile is None and not last:
            path, cfgfile = self.path_current, pathlib.Path('tmp.conf')
        elif cfgfile:
            path, cfgfile = self._load(cfgfile)

        if cfgfile is None and last:
            path, cfgfile = self._load(self.cfgfile_last)
            if cfgfile is None:
                raise ValueError('Configuration not found. Try last=False if the problem persist.')

        self.path, self.cfgfile = path, cfgfile
        self.run()

    def _load(self, cfgfile):
        """Help method to load cfgfiles."""
        cfgfile = pathlib.Path(cfgfile).absolute()
        path = cfgfile.absolute().parent
        if cfgfile.is_file() and path.is_dir():
            self._cfg.read(cfgfile)
            files.write_file(self.path_last, str(cfgfile), self.verbose)
            return path, cfgfile
        return None, None

    def write(self, fname):
        """Write a configuration file with current configuration."""

        cfg = ConfigParser(allow_no_value=True, converters={'lit': self.parse_literal})

        for section in self._SECTIONS:
            cfg.add_section(section)

        cfg.set('general', 'version', str(self.version)) #str(nsfds3.__version__))
        cfg.set('general', 'quiet', str(self.quiet))

        cfg.set('files', 'directory', str(self.files.directory))
        cfg.set('files', 'name', str(self.files.name))
        cfg.set('files', 'overwrite', str(self.files.overwrite))

        cfg.set('thermophysic', 'norm', str(self.tp.norm))
        cfg.set('thermophysic', 'rho0', str(self.tp.rho0))
        cfg.set('thermophysic', 't0', str(self.tp.T0 - self.tp.Tref))
        cfg.set('thermophysic', 'gamma', str(self.tp.gamma))

        cfg.set('geometry', 'free', str(self.geo.free))
        cfg.set('geometry', 'force', str(self.geo.force))
        cfg.set('geometry', 'file', str(self.geo.file))
        cfg.set('geometry', 'name', str(self.geo.name))
        cfg.set('geometry', 'kwargs', str(self.geo.kwargs))
        cfg.set('geometry', 'curvname', str(self.geo.curvname))
        cfg.set('geometry', 'bc', str(self.geo._bc))
        cfg.set('geometry', 'shape', str(self.geo._shape))
        cfg.set('geometry', 'origin', str(self.geo._origin))
        cfg.set('geometry', 'steps', str(self.geo._steps))
        cfg.set('geometry', 'flat', str(self.geo.flat))
        cfg.set('geometry', 'bz grid points', str(self.geo.bz_n))
        cfg.set('geometry', 'bz filter order', str(self.geo.bz_filter_order))
        cfg.set('geometry', 'bz stretch order', str(self.geo.bz_stretch_order))
        cfg.set('geometry', 'bz stretch factor', str(self.geo.bz_stretch_factor))

        cfg.set('sources', 'on', str(self.src.on))
        cfg.set('sources', 'origin', str(self.src._origin))
        cfg.set('sources', 'S0', str(self.src.S0))
        cfg.set('sources', 'Bx', str(self.src.Bx))
        cfg.set('sources', 'By', str(self.src.By))
        cfg.set('sources', 'Bz', str(self.src.Bz))
        cfg.set('sources', 'kx', str(self.src.kx))
        cfg.set('sources', 'ky', str(self.src.ky))
        cfg.set('sources', 'kz', str(self.src.kz))
        cfg.set('sources', 'k', str(self.src.k))
        cfg.set('sources', 'R', str(self.src.Rx))
        cfg.set('sources', 'evolution', str(self.src.evolution))

        cfg.set('flow', 'ftype', str(self.flw.ftype))
        cfg.set('flow', 'components', str(self.flw._components))

        cfg.set('probes', 'variables', str(self.prb.vars))
        cfg.set('probes', 'locations', str(self.prb._locs))

        cfg.set('solver', 'mp', str(self.sol.mp))
        cfg.set('solver', 'resume', str(self.sol.resume))
        cfg.set('solver', 'timings', str(self.sol.timings))
        cfg.set('solver', 'nt', str(self.sol.nt))
        cfg.set('solver', 'ns', str(self.sol.ns))
        cfg.set('solver', 'cfl', str(self.sol.cfl))
        cfg.set('solver', 'save fields', str(self.sol.save))
        cfg.set('solver', 'viscous fluxes', str(self.sol.vsc))
        cfg.set('solver', 'vorticity', str(self.sol.vrt))
        cfg.set('solver', 'shock capture', str(self.sol.cpt))
        cfg.set('solver', 'selective filter', str(self.sol.flt))
        cfg.set('solver', 'selective filter n-strength', str(self.sol.xnu_n))
        cfg.set('solver', 'selective filter 0-strength', str(self.sol.xnu_0))
        cfg.set('solver', 'nan_check', str(self.sol.nan_check))

        cfg.set('graphics', 'figures', str(self.gra.fig))
        cfg.set('graphics', 'probes', str(self.gra.prb))
        cfg.set('graphics', 'bz', str(self.gra.bz))
        cfg.set('graphics', 'bc', str(self.gra.bc))
        cfg.set('graphics', 'fps', str(self.gra.fps))

        fname = pathlib.Path(fname).with_suffix('.conf')
        if fname.parent:
            path = fname
        else:
            path = self.cfgfile.parent / fname

        with open(path, 'w') as fn:
            cfg.write(fn)

    def overwrite(self):
        """Overwrite configuration file."""
        self.write(self.cfgfile)

    @property
    def has_grid_backup(self):
        """Return whether a `CartesianGrid` or `CurvilinearGrid` already exist for this configuration or not."""
        try:
            cfg, _ = files.get_objects(self.files.directory, self.files.name)
        except FileNotFoundError:
            return False
        else:
            if self.geo == cfg.geo:
                return True
        return False

    def get_grid_backup(self):
        """Return existing `CartesianGrid` or `CurvilinearGrid` instance for this grid configuration
        if found, else return None."""
        if self.has_grid_backup:
            return files.get_objects(self.files.directory, self.files.name)[1]
        raise FileNotFoundError('Unable to load mesh file.')

    def run(self):
        """Run the parser."""

        # Create each section that does not exist in the configuration
        for section in self._SECTIONS:
            if not self._cfg.has_section(section):
                self._cfg.add_section(section)

        self.check_version()
        self._get_parameters()

        if not self.quiet and self.cfgfile is not None:
            print(f'\n[bold bright_cyan]{self.cfgfile}[bold bright_magenta] loaded.')
        elif not self.quiet:
            print(f'\n[bold bright_cyan]Default configuration[bold bright_magenta] loaded.')

    @property
    def version(self):
        """Version of configuration file."""
        return self._version

    @property
    def dt(self):
        """Time step.

        Note
        ----
        `dt` is a read only attribute. It is automatically updated if one of `geo.steps`,
        `sol.cfl`, `tp.c0` or `flw.components` attributes is modified.

        """
        if self.flw.ftype is not None:
            c = self.tp.c0 + max([abs(u) for u in self.flw.components])
        else:
            c = self.tp.c0
        return min(self.geo.steps) * self.sol.cfl / c

    @property
    def frequencies(self):
        """Calculate the minimum and maximum frequencies for the simulation.

        Returns
        -------
        tuple
            A tuple containing the minimum and maximum frequencies.
            The minimum frequency is calculated as 2 divided by the product of the time step and the number of time steps.
            The maximum frequency is calculated as the Courant-Friedrichs-Lewy (CFL) condition divided by 10 times the time step.
        """
        # To get two periods during the simulation
        fmin = 2 / (self.dt * self.sol.nt)
        # To get at least 10 points per wavelength
        fmax = self.sol.cfl / (10 * self.dt)
        return fmin, fmax

    @property
    def ndim(self):
        """Dimension of the grid."""
        return len(self.geo.shape)

    def _get_parameters(self):
        """Parse all simulation parameters."""

        CFG_GNL = self._cfg['general']
        self._version = CFG_GNL.get('version', self._version_base)
        self.quiet = CFG_GNL.getboolean('quiet', False)

        CFG_FILES = self._cfg['files']
        self.files = Files(path=self.path,
                           directory=self.path / pathlib.Path(CFG_FILES.get('directory', 'data/')),
                           name=pathlib.Path(CFG_FILES.get('name', self.cfgfile.stem)).with_suffix('.hdf5'),
                           overwrite=CFG_FILES.getboolean('overwrite', False),
                           verbose=self.verbose)

        CFG_GEO = self._cfg['geometry']
        self.geo = Geometry(shape=CFG_GEO.getlit('shape', (128, 96, 32)),
                            steps=CFG_GEO.getlit('steps', (1., 1., 1.)),
                            origin=CFG_GEO.getlit('origin', None),
                            bc=CFG_GEO.get('bc', 'WWWWWW').upper(),
                            flat=CFG_GEO.getlit('flat', None),
                            path=self.path,
                            file=CFG_GEO.getlit('file', None),
                            name=CFG_GEO.getlit('name', None),
                            kwargs=CFG_GEO.getlit('kwargs', None),
                            curvname=CFG_GEO.getlit('curvname', None),
                            bz_n=CFG_GEO.getint('bz grid points', 20),
                            bz_filter_order=CFG_GEO.getfloat('bz filter ordrer', 3.),
                            bz_stretch_factor=CFG_GEO.getfloat('bz stretch factor', 2.),
                            bz_stretch_order=CFG_GEO.getfloat('bz stretch order', 3.),
                            stencil=self.stencil,
                            force=CFG_GEO.getboolean('force', False),
                            free=CFG_GEO.getboolean('free', True))

        CFG_SOL = self._cfg['solver']
        self.sol = Solver(vsc=CFG_SOL.getboolean('viscous fluxes', True),
                          cpt=CFG_SOL.getboolean('shock capture', True),
                          vrt=CFG_SOL.getboolean('vorticity', True),
                          flt=CFG_SOL.getboolean('selective filter', True),
                          xnu_n=CFG_SOL.getfloat('selective filter n-strength', 0.2),
                          xnu_0=CFG_SOL.getfloat('selective filter 0-strength', 0.01),
                          nt=CFG_SOL.getint('nt', 50),
                          ns=CFG_SOL.getint('ns', 10),
                          cfl=CFG_SOL.getfloat('cfl', 0.5),
                          save=CFG_SOL.getboolean('save fields', True),
                          mp = CFG_SOL.getboolean('mp', True),
                          timings = CFG_SOL.getboolean('timings', False),
                          nan_check = CFG_SOL.getboolean('nan_check', False),
                          resume = CFG_SOL.getboolean('resume', False),
                          )

        CFG_SRC = self._cfg['sources']
        self.src = sources.SourceSet(shape=self.geo._shape,
                                     origin=CFG_SRC.getlit('origin', ((), )),
                                     S0=CFG_SRC.getlit('S0', ()),
                                     Bx=CFG_SRC.getlit('Bx', ()),
                                     By=CFG_SRC.getlit('By', ()),
                                     Bz=CFG_SRC.getlit('Bz', ()),
                                     kx=CFG_SRC.getlit('kx', ()),
                                     ky=CFG_SRC.getlit('ky', ()),
                                     kz=CFG_SRC.getlit('kz', ()),
                                     k=CFG_SRC.getlit('k', ()),
                                     Rx=CFG_SRC.getlit('R', ()),
                                     on=CFG_SRC.getlit('on', ()),
                                     evolution=CFG_SRC.getlit('evolution', ()),
                                     flat=self.geo.flat,
                                     )

        CFG_FLW = self._cfg['flow']
        self.flw = sources.Flow(shape=self.geo._shape,
                                ftype=CFG_FLW.get('type', None),
                                components=CFG_FLW.getlit('components', (0, ) * self.ndim),
                                flat=self.geo.flat,
                                )

        CFG_PRB = self._cfg['probes']
        self.prb = Probes(vars=CFG_PRB.getlit('variables', ()),
                          locs=CFG_PRB.getlit('locations', ()),
                          shape=self.geo._shape,
                          flat=self.geo.flat,
                          )

        CFG_THP = self._cfg['thermophysic']
        self.tp = Air(rho0=CFG_THP.getfloat('rho0', 1.2),
                      T0=CFG_THP.getfloat('T0', 20),
                      gamma=CFG_THP.getfloat('gamma', 1.4),
                      norm=CFG_THP.getboolean('norm', False),
                      )

        CFG_FIGS = self._cfg['graphics']
        self.gra = Graphics(fig=CFG_FIGS.getboolean('figures', True),
                            prb=CFG_FIGS.getboolean('probes', True),
                            bz=CFG_FIGS.getboolean('bz', True),
                            bc=CFG_FIGS.getboolean('bc', True),
                            fps=CFG_FIGS.getint('fps', 24)
                            )

    def check_version(self):
        """Check version of the configuration."""
        version = self._cfg['general'].get('version', self._version_base)
        version_ok = _parse_version(version) >= _parse_version(self._version_base)

        if not version_ok:
            print(f'[bold bright_magenta]Config file version must be >= {self._version_base}')
            sys.exit(1)

    @staticmethod
    def parse_literal(value):
        """Parse value."""
        if value.strip() == '':
            return None
        try:
            return ast.literal_eval(value)
        except ValueError:
            return value if value.lower() not in ('', 'no') else None

    def __eq__(self, other):
        if not isinstance(other, CfgSetup):
            raise ValueError(f'unsupported operand type(s) for +: {type(self).__name__} and {type(other).__name__}')

        return self.prb == other.prb and \
               self.tp == other.tp and \
               self.geo == other.geo and \
               self.sol == other.sol and \
               self.files.data_path == other.files.data_path and \
               self.dt == other.dt

    def __str__(self):

        size = misc.getsizeof(numpy.empty(self.geo.shape + (32, )))
        formatter = EngFormatter('B')

        s = "[System]"
        s += f"\n\t- Data path             : {self.files.data_path}"
        s += f"\n\t- Current path          : {self.path}"
        s += f"\n\t- Multiprocessing       : {self.sol.mp} [max {self.cpu_count} cpu(s)]"
        s += f"\n\t- Estimated ram used    : {formatter(size)}"
        s += str(self.sol)
        s += "\n[Time]"
        s += f"\n\t- Physical time    : {self.dt*self.sol.nt:.5e} s."
        s += f"\n\t- Time step        : dt={self.dt:.5e} s and nt={self.sol.nt}."
        s += "\n\t- Frequency range  : [{:.3f} - {:.3f}] Hz".format(*self.frequencies)
        s += str(self.tp)
        s += str(self.geo)
        s += str(self.src)
        s += str(self.flw)
        s += str(self.prb)
        return s

    def __repr__(self):
        return str(self)