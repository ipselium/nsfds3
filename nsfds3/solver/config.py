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
import pickle
import sys as _sys
import pathlib as _pathlib
from rich import print
from configparser import  ConfigParser
from pkg_resources import parse_version as _parse_version
from scipy import constants
from nsfds3.utils import files, misc
from nsfds3.solver import sources
from nsfds3.cpgrid import utils as cputils
from nsfds3.materials import Air


def _parse_int_tuple(input):
    if input.lower() not in CfgSetup._NONE:
        return tuple(int(k.strip()) for k in input[1:-1].split(','))
    return None


def _parse_float_tuple(input):
    if input.lower() not in CfgSetup._NONE:
        return tuple(float(k.strip()) for k in input[1:-1].split(','))
    return None


class CfgSetup:
    """ Setup configuration of the solver.

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
        data dir                    -> self.datadir = 'data/'
        data file                   -> self.datafile = 'tmp'
        timings                     -> self.timings = False
        quiet                       -> self.quiet = False
        cpu                         -> self.cpu = 1
        free                        -> self.free = True
        comp                        -> self.comp = False

        [thermophysics]
        norm                        -> self._norm = False
        rho0                        -> self.rho0 = 101325.0
        t0                          -> self.T0 - self._T_REF = 20.0
        gamma                       -> self.gamma = 1.4

        [geometry]
        geofile                     -> self.geofile = ''
        geoname                     -> self.geoname = None
        curvname                    -> self.curvname = None
        bc                          -> self.bc = 'WWWWWW'
        shape                       -> self.shape = (128, 96, 32)
        origin                      -> self.origin = None
        steps                       -> self.steps = (1., 1., 1.)
        flat                        -> self.flat = None
        bz grid points              -> self.bz_n = 20
        bz filter order             -> self.bz_filter_order = 3
        bz stretch order            -> self.bz_stretch_order = 3
        bz stretch factor           -> self.bz_stretch_factor = 2

        [initial pulses]
        on                          -> self.ics_on = False
        origins                     -> self.ics_origins = (),
        amplitudes                  -> self.ics_S0 = ()
        widths                      -> self.ics_B0 = ()

        [sources]
        on                          -> self.src_on = False
        origins                     -> self.src_origins = (),
        amplitudes                  -> self.src_S0 = ()
        widths                      -> self.src_B0 = ()
        evolutions                  -> self.src_evolutions = ()

        [flow]
        type                        -> self.flw_type = None
        components                  -> self.flw_components = (0, 0, 0)

        [solver]
        resume                      -> self.resume = False
        nt                          -> self.nt = 50
        ns                          -> self.ns = 10
        cfl                         -> self.CFL = 0.5
        probes                      -> self.prb = ()
        save fields                 -> self.save_fld = True
        viscous fluxes              -> self.vsc = True
        vorticity                   -> self.vrt = True
        shock capture               -> self.cpt = True
        selective filter            -> self.flt = True
        selective filter n-strength -> self.flt_xnu_n = 0.2
        selective filter 0-strength -> self.flt_xnu_0 = 0.01

        [figures]
        show figures                -> self.show_fig = True
        show probes                 -> self.show_prb = True
        show bz                     -> self.show_bz = True
        show bc                     -> self.show_bc = True
        fps                         -> self.fps = 24
    """

    _T_REF = constants.zero_Celsius     # Temperature 0°C
    _R_gp = constants.R                 # Molar gas constant
    _NONE = ('', 'no', 'No', 'none', 'None', None, 'False', 'false', False)
    _SECTIONS = ('general', 'thermophysic', 'geometry',
                 'sources', 'initial pulses',
                 'flow', 'solver', 'figures')

    def __init__(self, cfgfile=None, last=False, verbose=False):

        # Global attributes
        self.version_base = '0.1.0'
        self.stencil = 11
        self.verbose = verbose
        self.cpu_count = len(os.sched_getaffinity(0))
        self.path_nsfds3 = _pathlib.Path.home() / '.nsfds3'
        files.mkdir(self.path_nsfds3, self.verbose)
        self.path_last = self.path_nsfds3 / 'last'
        self.path_current = _pathlib.Path.cwd()
        self.cfgfile_last = _pathlib.Path(files.read_file(self.path_last, self.verbose))

        # Initialize configparser
        self._cfg = ConfigParser(allow_no_value=True,
                                 converters={'tuple_int': _parse_int_tuple,
                                             'tuple_float': _parse_float_tuple})

        # load the configuration file and parse all parameters
        self.load(cfgfile, last)

    def load(self, cfgfile=None, last=False):
        """ Load configuration file `cfgfile`. If file is not found, fallback to default configuration.

        Parameters
        ----------
        cfgfile: str, optional
            If None, CfgSetup takes defaults values.
            If a valid configuration file is provided, CfgSetup takes the value contained in this file.
        last: bool, optional
            If cfgfile is not provided or not found, try to load the last configuration used if it exists and if last is True.
        """
        if cfgfile is None and not last:
            path, cfgfile = self.path_current, _pathlib.Path('tmp.conf')
        elif cfgfile:
            path, cfgfile = self._load(cfgfile)

        if cfgfile is None and last:
            path, cfgfile = self._load(self.cfgfile_last)
            if cfgfile is None:
                raise ValueError('Configuration not found. Try last=False if the problem persist.')

        self.path, self.cfgfile = path, cfgfile
        self.run()

    def _load(self, cfgfile):
        """ Help method to load cfgfiles. """
        cfgfile = _pathlib.Path(cfgfile).absolute()
        path = cfgfile.absolute().parent
        if cfgfile.is_file() and path.is_dir():
            self._cfg.read(cfgfile)
            files.write_file(self.path_last, str(cfgfile), self.verbose)
            return path, cfgfile
        return None, None

    def write(self, fname):
        """ Write a configuration file with current configuration. """

        self._cfg.set('general', 'version', str(self.version)) #str(nsfds3.__version__))
        self._cfg.set('general', 'data dir', str(self.datadir))
        self._cfg.set('general', 'data file', str(self.datafile))
        self._cfg.set('general', 'timings', str(self.timings))
        self._cfg.set('general', 'quiet', str(self.quiet))
        self._cfg.set('general', 'cpu', str(self.cpu))
        self._cfg.set('general', 'free', str(self.free))
        self._cfg.set('general', 'comp', str(self.comp))

        self._cfg.set('thermophysic', 'norm', str(self._norm))
        self._cfg.set('thermophysic', 'rho0', str(self.rho0))
        self._cfg.set('thermophysic', 't0', str(self.T0 - self._T_REF))
        self._cfg.set('thermophysic', 'gamma', str(self.gamma))

        self._cfg.set('geometry', 'geofile', str(self.geofile))
        self._cfg.set('geometry', 'geoname', str(self.geoname))
        self._cfg.set('geometry', 'curvname', str(self.curvname))
        self._cfg.set('geometry', 'bc', str(self.bc))
        self._cfg.set('geometry', 'shape', str(self.shape))
        self._cfg.set('geometry', 'origin', str(self.origin))
        self._cfg.set('geometry', 'steps', str(self.steps))
        self._cfg.set('geometry', 'flat', str(self.flat))
        self._cfg.set('geometry', 'bz grid points', str(self.bz_n))
        self._cfg.set('geometry', 'bz filter order', str(self.bz_filter_order))
        self._cfg.set('geometry', 'bz stretch order', str(self.bz_stretch_order))
        self._cfg.set('geometry', 'bz stretch factor', str(self.bz_stretch_factor))

        self._cfg.set('initial pulses', 'on', str(self.ics_on))
        self._cfg.set('initial pulses', 'origins', str(self.ics_origins))
        self._cfg.set('initial pulses', 'amplitudes', str(self.ics_S0))
        self._cfg.set('initial pulses', 'widths', str(self.ics_B0))

        self._cfg.set('sources', 'on', str(self.src_on))
        self._cfg.set('sources', 'origins', str(self.src_origins))
        self._cfg.set('sources', 'amplitudes', str(self.src_S0))
        self._cfg.set('sources', 'widths', str(self.src_B0))
        self._cfg.set('sources', 'evolutions', str(self.src_evolutions))

        self._cfg.set('flow', 'type', str(self.flw_type))
        self._cfg.set('flow', 'components', str(self.flw_components))

        self._cfg.set('solver', 'resume', str(self.resume))
        self._cfg.set('solver', 'nt', str(self.nt))
        self._cfg.set('solver', 'ns', str(self.ns))
        self._cfg.set('solver', 'cfl', str(self.CFL))
        self._cfg.set('solver', 'probes', str(self.prb))
        self._cfg.set('solver', 'save fields', str(self.save_fld))
        self._cfg.set('solver', 'viscous fluxes', str(self.vsc))
        self._cfg.set('solver', 'vorticity', str(self.vrt))
        self._cfg.set('solver', 'shock capture', str(self.cpt))
        self._cfg.set('solver', 'selective filter', str(self.flt))
        self._cfg.set('solver', 'selective filter n-strength', str(self.flt_xnu_n))
        self._cfg.set('solver', 'selective filter 0-strength ', str(self.flt_xnu_0))

        self._cfg.set('figures', 'show figures', str(self.show_fig))
        self._cfg.set('figures', 'show probes', str(self.show_prb))
        self._cfg.set('figures', 'show bz', str(self.show_bz))
        self._cfg.set('figures', 'show bc', str(self.show_bc))
        self._cfg.set('figures', 'fps', str(self.fps))

        fname = fname if fname.endswith('.conf') else f"{fname}.conf"
        with open(self.datadir / fname, 'w') as fn:
            self._cfg.write(fn)

    def has_same_grid_configuration_as(self, other):
        """ Report whether self and other have same grid configuration or not. """
        if not isinstance(other, CfgSetup):
            raise ValueError('Can only compare CfgSetup together')

        attrs = ['shape', 'steps', 'origin', 'bc', 'obstacles', 
                 'bz_n', "bz_stretch_factor", "bz_stretch_order", 
                 "stencil", "free", "curvfunc"]

        for attr in attrs:

            a1 = getattr(self, attr, None)
            a2 = getattr(other, attr, None)

            if callable(a1) and callable(a2):
                if a1.__name__ != a2.__name__:
                    return False
            elif a1 != a2:
                return False

        return True

    def get_grid_configuration(self):
        """ Return arg and kwargs needed to instanciate `CartesianGrid` or `CurvilinearGrid`. """
        args = self.shape, self.steps
        kwargs = {'origin': self.origin,
                  'bc': self.bc,
                  'obstacles': self.obstacles,
                  'bz_n': self.bz_n,
                  'bz_stretch_factor': self.bz_stretch_factor,
                  'bz_stretch_order': self.bz_stretch_order,
                  'stencil': self.stencil,
                  'free': self.free}
        if self.curvfunc:
            kwargs['curvfunc'] = self.curvfunc
        return args, kwargs

    def get_grid_backup(self):
        """ Return existing `CartesianGrid` or `Curvilinear` object for this grid configuration 
        if found, else return None. """
        cfg, msh = files.get_objects(self.datadir, self.datafile)
        if self.has_same_grid_configuration_as(cfg):
            return msh
        return None

    def run(self):
        """ Run the parser. """

        # Create each section that does not exist in the configuration
        for section in self._SECTIONS:
            if not self._cfg.has_section(section):
                self._cfg.add_section(section)

        self._check_version()
        self._get_parameters()
        if self.flat and len(self.shape) == 3:
            self._3d_to_2d()

        if not self.quiet and self.cfgfile is not None:
            print(f'\n[bold red]{self.cfgfile}[/] loaded.\n')
        elif not self.quiet:
            print(f'\n[bold red]Default configuration[/] loaded.\n')

    @property
    def version(self):
        """ Version of configuration file. """
        return self._version

    @property
    def datapath(self):
        """ Absolute path to datafile.

        Note
        ----
        datapath is read only. It cannot be set directly. Instead, set `datadir` and `datafile`.
        """
        return self.datadir / self.datafile

    @property
    def datafile(self):
        """ Data file name used to save fields. """
        return self._datafile

    @datafile.setter
    def datafile(self, value):
        if not isinstance(value, str):
            raise ValueError('datafile: str expected')
        self._datafile = _pathlib.Path(value).with_suffix('.hdf5')

    @property
    def datadir(self):
        """ Directory where to save data files.

        Note
        ----
        If directory does not exist, create it.
        """
        return self._datadir

    @datadir.setter
    def datadir(self, value):
        if not isinstance(value, (str, _pathlib.Path)):
            raise ValueError('datadir: str or pathlib.Path expected')
        if isinstance(value, str):
            value = _pathlib.Path(value)
        self._datadir = self.path / value
        files.mkdir(self.datadir, self.verbose)

    @property
    def nt(self):
        """ Number of time iterations.

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
        self._adjust_nt()

    @property
    def ns(self):
        """ Field backup frequency.

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
        self._adjust_nt()

    @property
    def dt(self):
        """ Time step.

        Note
        ----
        `dt` is a read only attribute. It is automatically updated if one of `steps`,
        `CFL`, `c0` or `flw_components` attributes is modified.

        """
        if self.flw_type not in self._NONE:
            c = self.tp.c0 + max([abs(u) for u in self.flw_components])
        else:
            c = self.tp.c0
        return min(self.steps) * self.CFL / c

    @property
    def shape(self):
        """ Shape of the computation domain. """
        return self._shape

    @shape.setter
    def shape(self, value):
        value = cputils.parse_shape(value)
        self._shape = value

    @property
    def steps(self):
        """ Spatial steps (dx, dy[, dz]).

        Note
        ----
        If `steps` is modified, the time step `dt` is modified too.
        """
        return self._steps

    @steps.setter
    def steps(self, value):
        value = cputils.parse_steps(self.shape, value)
        self._steps = value

    @property
    def origin(self):
        """ Origin of the computation domain. """
        return self._origin

    @origin.setter
    def origin(self, value):
        value = cputils.parse_origin(self.shape, value, self.bc, self.bz_n)
        self._origin = value

    @property
    def bc(self):
        """ Boundary condition of the computation domain. """
        return self._bc

    @bc.setter
    def bc(self, value):
        value = cputils.parse_bc(self.shape, value)
        self._bc = value

    @property
    def flat(self):
        """ Describe how a 3d configuration parameters are converted to a 2d parameters.

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
    def geofile(self):
        """ Name of the file in which to search for functions `geoname`,
        `curvname`, and functions used for sources.
        """
        return self._geofile

    @geofile.setter
    def geofile(self, value):
        self._geofile = value
        self._update_obstacles()
        self._update_curvilinear_transformation()

    @property
    def geoname(self):
        """ Name of the function to be used to set up the `Obstacle` arrangement. """
        return self._geoname

    @geoname.setter
    def geoname(self, value):
        self._geoname = value
        self._update_obstacles()

    @property
    def curvname(self):
        """ Name of the function to be used to set up the curvilinear transformation. """
        return self._curvname

    @curvname.setter
    def curvname(self, value):
        self._curvname = value
        self._update_curvilinear_transformation()

    @property
    def norm(self):
        """ Report whether thermophysic variables are normalized.

        Note
        ----
        If `norm` is modified, all thermophysic variables are automatically updated.
        """
        return self._norm

    @norm.setter
    def norm(self, value):
        if not isinstance(value, bool):
            raise ValueError('norm: boolean expected')
        if value:
            self._thp_norm()
        else:
            self._thp_fixed()

    @property
    def zeros(self):
        """ Return shape-size tuple of zeros. """
        return (0, ) * len(self.shape)

    @property
    def ics_origins(self):
        """ Origins of the initial pressure pulses. """
        return self._ics_origins

    @ics_origins.setter
    def ics_origins(self, value):
        if not isinstance(value, tuple):
            raise ValueError('ics_origin: tuple expected')
        self._ics_origins = value
        self._update_ics()

    @property
    def ics_S0(self):
        """ Amplitudes of the initial pressure pulses. """
        return self._ics_S0

    @ics_S0.setter
    def ics_S0(self, value):
        if not isinstance(value, tuple):
            raise ValueError('ics_S0: tuple expected')
        self._ics_S0 = value
        self._update_ics()

    @property
    def ics_B0(self):
        """ Widths of the initial pressure pulses. """
        return self._ics_B0

    @ics_B0.setter
    def ics_B0(self, value):
        if not isinstance(value, tuple):
            raise ValueError('ics_B0: tuple expected')
        self._ics_B0 = value
        self._update_ics()

    @property
    def src_origins(self):
        """ Origins of the sources. """
        return self._src_origins

    @src_origins.setter
    def src_origins(self, value):
        if not isinstance(value, tuple):
            raise ValueError('src_origin: tuple expected')
        self._src_origins = value
        self._update_src()

    @property
    def src_S0(self):
        """ Amplitudes of the sources. """
        return self._src_S0

    @src_S0.setter
    def src_S0(self, value):
        if not isinstance(value, tuple):
            raise ValueError('src_S0: tuple expected')
        self._src_S0 = value
        self._update_src()

    @property
    def src_B0(self):
        """ Widths of the sources. """
        return self._src_B0

    @src_B0.setter
    def src_B0(self, value):
        if not isinstance(value, tuple):
            raise ValueError('src_B0: tuple expected')
        self._src_B0 = value
        self._update_src()

    @property
    def src_evolutions(self):
        """ Time evolutions of the sources. """
        return self._src_B0

    @src_evolutions.setter
    def src_evolutions(self, value):
        if not isinstance(value, tuple):
            raise ValueError('src_evolution: tuple expected')
        self._src_evolutions = value
        self._update_src()

    @property
    def flw_type(self):
        """ Type of mean flow. """
        return self._flw_type

    @flw_type.setter
    def flw_type(self, value):
        self._check_flow(value, self.flw_components)
        self._flw_type = value

    @property
    def flw_components(self):
        """ Components of the mean flow.

        Note
        ----
        If `flw_component` is modified, the time step `dt` is modified too if `flw_type` is set to an actual flow.
        """
        return self._flw_components

    @flw_components.setter
    def flw_components(self, value):
        self._check_flow(self.flw_type, value)
        self._flw_components = value

    @property
    def prb(self):
        """ Locations of the probes. """
        return self._prb

    @prb.setter
    def prb(self, value):
        self._prb = self._parse_probes(value)

    @property
    def flt_xnu_n(self):
        """ Selective filter strength.

        Note
        ----
        Must be between 0 and 1
        """
        return self._flt_xnu_n

    @flt_xnu_n.setter
    def flt_xnu_n(self, value):
        self._check_filter(value)
        self._flt_xnu_n = value

    @property
    def flt_xnu_0(self):
        """ Selective filter strength for points close to boundaries.

        Note
        ----
        Must be between 0 and 1
        """
        return self._flt_xnu_0

    @flt_xnu_0.setter
    def flt_xnu_0(self, value):
        self._check_filter(value)
        self._flt_xnu_0 = value

    def _get_parameters(self):
        """ Parse all simulation parameters. """

        GNL = self._cfg['general']
        self._version = GNL.get('version', self.version_base)
        self._datadir = self.path / _pathlib.Path(GNL.get('data dir', 'data/'))
        self._datafile = _pathlib.Path(GNL.get('data file', self.cfgfile.stem)).with_suffix('.hdf5')
        self.timings = GNL.getboolean('timings', False)
        self.quiet = GNL.getboolean('quiet', False)
        self.cpu = GNL.getint('cpu', int(self.cpu_count / 2))
        self.free = GNL.getboolean('free', True)
        self.comp = GNL.getboolean('comp', False)
        files.mkdir(self.datadir, self.verbose)

        GEO = self._cfg['geometry']
        self.bz_n = GEO.getint('bz grid points', 20)
        self.bz_filter_order = GEO.getfloat('bz filter ordrer', 3.)
        self.bz_stretch_order = GEO.getfloat('bz stretch order', 3.)
        self.bz_stretch_factor = GEO.getfloat('bz stretch factor', 2.)
        self._shape = GEO.gettuple_int('shape', (128, 96, 32))
        self._shape = cputils.parse_shape(self._shape)
        self._steps = GEO.gettuple_float('steps', (1., 1., 1.))
        self._steps = cputils.parse_steps(self._shape, self._steps)
        self._bc = GEO.get('bc', 'WWWWWW').upper()
        self._bc = cputils.parse_bc(self._shape, self._bc)
        self._origin = GEO.gettuple_int('origin', None)
        self._origin = cputils.parse_origin(self._shape, self._origin, self._bc, self.bz_n)
        self._flat = GEO.gettuple_int('flat', None)
        self._geofile = GEO.get('geofile', '')
        self._geoname = GEO.get('geoname', None)
        self._curvname = GEO.get('curvname', None)
        self._update_obstacles()
        self._update_curvilinear_transformation()

        SOL = self._cfg['solver']
        self.resume = SOL.getboolean('resume', False)
        self._nt = SOL.getint('nt', 50)
        self._ns = SOL.getint('ns', 10)
        self.CFL = SOL.getfloat('cfl', 0.5)
        self._prb = ast.literal_eval(SOL.get('probes', '()'))
        self._prb = self._parse_probes(self._prb)
        self.save_fld = SOL.getboolean('save fields', True)
        self.vsc = SOL.getboolean('viscous fluxes', True)
        self.vrt = SOL.getboolean('vorticity', True)
        self.cpt = SOL.getboolean('shock capture', True)
        self.flt = SOL.getboolean('selective filter', True)
        self._flt_xnu_n = SOL.getfloat('selective filter n-strength', 0.2)
        self._flt_xnu_0 = SOL.getfloat('selective filter 0-strength', 0.01)
        self.it = 0
        self._check_filter(self._flt_xnu_n)
        self._check_filter(self._flt_xnu_0)
        self._adjust_nt()

        ICS = self._cfg['initial pulses']
        self.ics_on = ICS.getboolean('on', False)
        self._ics_origins = ast.literal_eval(ICS.get('origins', '(), '))
        self._ics_S0 = ast.literal_eval(ICS.get('amplitudes', '()'))
        self._ics_B0 = ast.literal_eval(ICS.get('widths', '()'))
        self._update_ics()

        SRC = self._cfg['sources']
        self.src_on = SRC.getboolean('on', False)
        self._src_origins = ast.literal_eval(SRC.get('origins', '(), '))
        self._src_S0 = ast.literal_eval(SRC.get('amplitudes', '()'))
        self._src_B0 = ast.literal_eval(SRC.get('widths', '()'))
        self._src_evolutions = ast.literal_eval(SRC.get('evolutions', '()'))
        self._update_src()

        FLW = self._cfg['flow']
        self._flw_type = FLW.get('type', 'None').lower()
        self._flw_components = FLW.gettuple_float('components', self.zeros)
        self._check_flow(self.flw_type, self.flw_components)

        THP = self._cfg['thermophysic']
        norm = THP.getboolean('norm', False)
        rho0 = THP.getfloat('rho0', 1.2)
        T0 = THP.getfloat('T0', 20)
        gamma = THP.getfloat('gamma', 1.4)
        self.tp = Air(rho0, T0, gamma, norm)

        FIGS = self._cfg['figures']
        self.show_fig = FIGS.getboolean('show figures', True)
        self.show_prb = FIGS.getboolean('show probes', True)
        self.show_bz = FIGS.getboolean('show bz', True)
        self.show_bc = FIGS.getboolean('show bc', True)
        self.fps = FIGS.getint('fps', 24)


    @staticmethod
    def to_2d_tuple(var, ax):
        """ Return a 2d version of the tuple `var` removing the ax-th value. """
        return tuple(s for i, s in enumerate(var) if i != ax)

    def _3d_to_2d(self):

        self._check_flat(self.flat)
        ax, idx = self.flat
        self._shape = self.to_2d_tuple(self.shape, ax)
        self._steps = self.to_2d_tuple(self.steps, ax)
        self._origin = self.to_2d_tuple(self.origin, ax)
        self._flw_components = self.to_2d_tuple(self.flw_components, ax)
        self._bc = ''.join(bc for i, bc in enumerate(self.bc) if i not in [2*ax, 2*ax + 1])
        self.obstacles = [obs.flatten(ax) for obs in self.obstacles if idx in obs.rn[ax]]

        for s in self.ics:
            s.origin = self.to_2d_tuple(s.origin, ax)

        for s in self.src:
            s.origin = self.to_2d_tuple(s.origin, ax)

        if self.prb:
            self._prb = [[c for i, c in enumerate(prb) if i != ax] for prb in self.prb]

    def _adjust_nt(self):
        """ Adjust the number of time iterations `nt` to be a multiple of `ns`. """
        if self._nt % self._ns:
            self._nt -= self._nt % self._ns

    def _parse_probes(self, locations):
        if not isinstance(locations, tuple):
            raise ValueError('Probes: tuple expected')

        if locations and not any(isinstance(p, (tuple, list)) for p in self.prb):
            locations = locations,

        for loc in locations:
            if len(loc) != len(self.shape):
                raise ValueError(f'Probes: tuple of length {len(self.shape)} expected')
            if any(not 0 <= c < s for c, s in zip(loc, self.shape)):
                raise ValueError('Probes: out of bounds')

        return locations

    def _check_version(self):
        """ Check version of the configuration."""
        version = self._cfg['general'].get('version', self.version_base)
        version_ok = _parse_version(version) >= _parse_version(self.version_base)

        if not version_ok:
            print(f'Config file version must be >= {self.version_base}')
            _sys.exit(1)

    def _check_filter(self, strength):
        """ Check that the strength of the filter is set correctly. """
        if not isinstance(strength, float):
            raise ValueError('Filter: float expected')

        if not 0 <= strength <= 1:
            raise ValueError('Filter: strength must be between 0 and 1')

    def _check_flow(self, flw_type, flw_components):
        """ Check that flow type and components are set correctly. """
        if not isinstance(flw_components, tuple):
            raise ValueError('Mean flow: tuple expected for components')

        if not isinstance(flw_type, (str, type(None))):
            raise ValueError('Mean flow: str or NoneType expected for flow type')

        if len(flw_components) != len(self.shape):
            raise ValueError(f'Mean flow: component must be {len(self.shape)}d')

        if flw_type not in ('mean flow', ) + self._NONE:
            raise ValueError("Flow: must be 'mean flow' or None")

    def _check_flat(self, flat):
        """ Check that flat is consistent. """
        if not isinstance(flat, (tuple, type(None))):
            raise ValueError('flat: tuple of None expected')

        if flat is not None:
            if len(flat) != 2:
                raise ValueError('flat: length 2 expected (axis, location)')

            flat_ax, flat_idx = self.flat

            if flat_ax not in range(3):
                raise ValueError('flat: element 0 (axis) must be 0, 1, or 2')

            if flat_idx not in range(self.shape[flat_ax]):
                raise ValueError('flat: element 1 (index) must be in the domain')

    def _check_ics(self):
        """ Check that initial conditions are consistents. """
        if not any(isinstance(o, (tuple, list)) for o in self.ics_origins):
            self._ics_origins = self._ics_origins,

        if isinstance(self.ics_S0, (int, float)):
            self._ics_S0 = self._ics_S0,

        if isinstance(self.ics_B0, (int, float)):
            self._ics_B0 = self._ics_B0,

        if not all([len(o) == len(self.shape) for o in self.ics_origins]):
            raise ValueError(f'ics_origin: each element must be {len(self.shape)}')

    def _check_src(self):
        """ Check that sources are consistents. """
        if not any(isinstance(o, (tuple, list)) for o in self.src_origins):
            self._src_origins = self._src_origins,

        if isinstance(self.src_S0, (int, float)):
            self._src_S0 = self._src_S0,

        if isinstance(self.src_B0, (int, float)):
            self._src_B0 = self._src_B0,

        if isinstance(self.src_evolutions, (int, float, str)):
            self._src_B0 = self._src_B0,

        if not all([len(o) == len(self.shape) for o in self.src_origins]):
            raise ValueError(f'src_origin: each element must be {len(self.shape)}')

    def _update_ics(self):
        """ Update initial conditions parameters. """
        self.ics = []
        if self.ics_on:
            self._check_ics()
            for o, s, b in zip(self.ics_origins, self.ics_S0, self.ics_B0):
                self.ics.append(sources.Pulse(o, s, b))

    def _update_src(self):
        """ Update sources parameters. """
        self.src = []
        if self.src_on :
            self._check_src()
            for o, s, b, e in zip(self.src_origins, self.src_S0, self.src_B0, self.src_evolutions):
                self.src.append(sources.Monopole(o, s, b, e))

    def _update_obstacles(self):
        if self.geoname not in self._NONE:
            geofile = '' if self.geofile is None else self.geofile
            self.obstacles = files.get_func(self.path / geofile, self.geoname)
            if self.obstacles is not None:
                self.obstacles = self.obstacles(self.shape)
        else:
            self.obstacles = None

        self.obstacles = [] if self.obstacles is None else self.obstacles

    def _update_curvilinear_transformation(self):
        if self.curvname not in self._NONE:
            geofile = '' if self.geofile is None else self.geofile
            self.curvfunc = files.get_func(self.path / geofile, self.curvname)
        else:
            self.curvfunc = None

    def __eq__(self, other):
        if not isinstance(other, CfgSetup):
            raise ValueError(f'unsupported operand type(s) for +: {type(self).__name__} and {type(other).__name__}')

        return misc.deep_equals(self, other)

    def __str__(self):

        # Solver
        s = "* Solver : \n"
        s += f"\t- Viscous fluxes      : {self.vsc}\n"
        s += f"\t- Selective filter    : {self.flt} [nu_0={self.flt_xnu_0}, nu_n={self.flt_xnu_n}]\n"
        s += f"\t- Shock capture       : {self.cpt}\n\n"

        # Thermophysics
        s += "* Thermophysic : \n"
        for tp in self.tp.__repr__().split('\n'):
            s += f"\t{tp}\n"

        # Grid
        s += f"* {'CartesianGrid' if not self.curvfunc else 'CurvilinearGrid'} : \n"
        s += f"\t- Grid                : {'x'.join(str(n) for n in self.shape)} points grid\n"
        s += f'\t- boundary conditions : {self.bc}\n'
        s += f"\t- Spatial step        : ({', '.join(str(n) for n in self.steps)})\n"
        s += f"\t- Origin              : ({', '.join(str(n) for n in self.origin)})\n"
        if 'A' in self.bc:
            s += f'\t- Buffer zone         : {self.bz_n} grid points\n'
        if self.obstacles:
            s += f"\t- Obstacles           : {self.obstacles}\n"
        if hasattr(self, 'curvfunc'):
            s += f"\t- Curvilinear         : {self.curvfunc}\n"

        # Time
        s += f"* Time :\n"
        s += f"\t- Physical time           : {self.dt*self.nt:.5e} s.\n"
        s += f"\t- Time step               : dt={self.dt:.5e} s and nt={self.nt}.\n\n"

        # Sources
        if self.obstacles:
            wall_source = any('V' in o.bc for o in self.obstacles)
        else:
            wall_source = False

        if self.src or self.ics or wall_source:
            s += f"* Sources :\n"

        if self.ics:
            for ic in self.ics:
                s += f"\t- {ic}.\n"

        if self.src:
            for source in self.src:
                s += f"\t- {source}.\n"

        if wall_source:
            s += f"\t- Wall source setup"

        if self.flw_type not in self._NONE:
            s += f"\t* flow         : {self.flw_type} {self.flw_components}.\n"

        return s

    def __repr__(self):
        return self.__str__()