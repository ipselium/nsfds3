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
-----------

This module contains the :py:class:`CfgSetup` that read the configuration file
and set all simulation parameters.

Example
-------

::

    from nsfds3.init import CfgSetup

    cfg = CfgSetup()

-----------
"""
import ast
import time as _time
import sys as _sys
import pathlib as _pathlib
import configparser as _configparser
from pkg_resources import parse_version as _parse_version
from nsfds3.utils import files
from nsfds3.solver.sources import Pulse, Monopole


def _parse_int_tuple(input):
    if input.lower() not in [None, 'none']:
        return tuple(int(k.strip()) for k in input[1:-1].split(','))
    return None


def _parse_float_tuple(input):
    if input.lower() not in [None, 'none']:
        return tuple(float(k.strip()) for k in input[1:-1].split(','))
    return None


def mkdir(directory):
    """ Check if dir exists. If not, create it."""
    if not directory.is_dir():
        directory.mkdir()
        print("Create directory :", directory)
        _time.sleep(0.5)


class CfgSetup:
    """ Handle configuration file. """

    _T_REF = 273.15
    _NONE = ('', 'no', 'No', 'none', 'None', None)
    _SECTIONS = ('general', 'thermophysic', 'geometry',
                 'sources', 'initial pulses',
                 'flow', 'solver', 'figures')

    def __init__(self, cfgfile=None):

        # Minimal version of the config file
        self.base_version = '0.1.0'
        self.cfgfile = cfgfile
        self.path = _pathlib.Path.cwd()
        self.stencil = 11

        # Create config parser
        self._cfg = _configparser.ConfigParser(allow_no_value=True,
                                               converters={'tuple_int': _parse_int_tuple,
                                                           'tuple_float': _parse_float_tuple})

        # Load config file
        if isinstance(self.cfgfile, (_pathlib.Path, str)):
            cfgfile = _pathlib.Path(self.cfgfile)
            path = cfgfile.absolute().parent
            if not cfgfile.is_file() or not path.is_dir():
                print('Wrong configuretion file - Fallback to default configuration')
            else:
                self._cfg.read(self.cfgfile)

        # Create each section that does not exist
        for section in self._SECTIONS:
            if not self._cfg.has_section(section):
                self._cfg.add_section(section)

        # Parse arguments
        self.check_version()
        self.run()

    def check_version(self):
        """ Check version of the config file. Overwrite it if too old. """
        version = self._cfg['general'].get('version', self.base_version)
        version_ok = _parse_version(version) >= _parse_version(self.base_version)

        if not version_ok:
            print(f'Config file version must be >= {self.base_version}')
            _sys.exit(1)

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
        self._cfg.set('thermophysic', 'p0', str(self.p0))
        self._cfg.set('thermophysic', 't0', str(self.T0 - self._T_REF))
        self._cfg.set('thermophysic', 'gamma', str(self.gamma))
        self._cfg.set('thermophysic', 'prandtl', str(self.prandtl))

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
        with open(self.path / fname, 'w') as fn:
            self._cfg.write(fn)

    def run(self):
        """ Run configuration. """

        self._gnl()
        self._thp()
        self._geo()
        self._sol()
        self._src()
        self._ics()
        self._flw()
        self._figs()

        if self.flat and len(self.shape) == 3:
            self._3d_to_2d()

        c = self.c0 + max([abs(u) for u in self.flw_components])
        self.dt = min(self.steps) * self.CFL / c

    def _gnl(self):
        """ Get general parameters of nsfds3. """
        GNL = self._cfg['general']
        self.version = GNL.get('version', self.base_version)
        self.datadir = _pathlib.Path(GNL.get('data dir', 'data/'))
        s = GNL.get('data file', 'tmp')
        self.datafile = s if s.endswith('.hdf5') else f"{s}.hdf5"
        self.datapath = self.datadir / self.datafile
        self.timings = GNL.getboolean('timings', False)
        self.quiet = GNL.getboolean('quiet', False)
        self.cpu = GNL.getint('cpu', 1)
        self.free = GNL.getboolean('free', True)
        self.comp = GNL.getboolean('comp', False)

        mkdir(self.datadir)

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        if not isinstance(value, bool):
            raise ValueError('norm must be boolean')
        if value:
            self._thp_norm()
        else:
            self._thp_fixed()

    def _thp(self):
        """ Get thermophysical parameters. """
        THP = self._cfg['thermophysic']
        self._norm = THP.getboolean('norm', False)
        self.Ssu = 110.4        # Sutherland constant
        self.gamma = THP.getfloat('gamma', 1.4)
        self.CV = 717.5
        self.CP = self.CV * self.gamma
        self.prandtl = THP.getfloat('prandtl', 0.7)
        self.mu0 = 0.00001716                                       # Dynamic viscosity at T = 0 deg. C

        if self.norm:
            self._thp_norm()
        else:
            self._thp_fixed()

        if self.c0 < 1:
            raise ValueError('c0 must be >= 1')

    def _thp_fixed(self):
        THP = self._cfg['thermophysic']
        self.T0 = self._T_REF + THP.getfloat('t0', 20.0)
        self.p0 = THP.getfloat('p0', 101325.0)
        self.rho0 = self.p0 / (self.T0 * (self.CP - self.CV))
        self.c0 = (self.gamma * self.p0 / self.rho0)**0.5
        self.mu = (self.mu0 * (self.T0 / self._T_REF)**(3. / 2.) *
                   (self._T_REF + self.Ssu) / (self.T0 + self.Ssu))   # Dynamic viscosity at T0
        self.nu = self.mu / self.rho0                                 # Kinematic viscosity at T0

    def _thp_norm(self):

        self.T0 = 299.8189
        self.rho0 = 1
        self.c0 = 1
        self.p0 = self.rho0 * self.c0**2 / self.gamma
        self.mu = (self.mu0 * (self.T0 / self._T_REF)**(3. / 2.) *
                   (self._T_REF + self.Ssu) / (self.T0 + self.Ssu))   # Dynamic viscosity at T0
        self.nu = self.mu / self.rho0                                 # Kinematic viscosity at T0

    def _geo(self):
        """ Get geometry parameters. """
        GEO = self._cfg['geometry']

        # Grid
        self.bc = GEO.get('bc', 'WWWWWW').upper()
        self.shape = GEO.gettuple_int('shape', (256, 256, 256))
        self.origin = GEO.gettuple_int('origin', None)
        self.steps = GEO.gettuple_float('steps', (1., 1., 1.))
        self.flat = GEO.gettuple_int('flat', None)

        # Buffer zone
        self.bz_n = GEO.getint('bz grid points', 20)
        self.bz_filter_order = GEO.getfloat('bz filter ordrer', 3.)
        self.bz_stretch_order = GEO.getfloat('bz stretch order', 3.)
        self.bz_stretch_factor = GEO.getfloat('bz stretch factor', 2.)


        # Mesh type and geometry
        self.geofile = GEO.get('geofile', '')
        self.geoname = GEO.get('geoname', None)
        self.curvname = GEO.get('curvname', None)

        if len(self.shape) != len(self.steps) or 2 * len(self.shape) != len(self.bc):
            raise ValueError('shape, steps and bc must have coherent dims.')

        # Geometry
        if self.geoname not in self._NONE:
            self.obstacles = files.get_func(self.path / self.geofile, self.geoname)
            if self.obstacles:
                self.obstacles = self.obstacles(self.shape, self.stencil)
        else:
            self.obstacles = None

        # Curvilinear mesh
        if self.curvname not in self._NONE:
            self.curvilinear_func = files.get_func(self.path / self.geofile, self.curvname)
        else:
            self.curvilinear_func = None

    def _ics(self):
        """ Get initial conditions. """
        ICS = self._cfg['initial pulses']
        self.ics_on = ICS.getboolean('on', False)
        self.ics_origins = ast.literal_eval(ICS.get('origins', '()'))
        self.ics_S0 = ast.literal_eval(ICS.get('amplitudes', '()'))
        self.ics_B0 = ast.literal_eval(ICS.get('widths', '()'))

        self.ics = []
        if self.ics_on:
            if isinstance(self.ics_B0, (float, int)):
                self.ics.append(Pulse(self.ics_origins, self.ics_S0, self.ics_B0))
            else:
                for o, s, b in zip(self.ics_origins, self.ics_S0, self.ics_B0):
                    self.ics.append(Pulse(o, s, b))

    def _src(self):
        """ Get sources. """

        SRC = self._cfg['sources']
        self.src_on = SRC.getboolean('on', False)
        self.src_origins = ast.literal_eval(SRC.get('origins', '()'))
        self.src_S0 = ast.literal_eval(SRC.get('amplitudes', '()'))
        self.src_B0 = ast.literal_eval(SRC.get('widths', '()'))
        self.src_evolutions = ast.literal_eval(SRC.get('evolutions', '()'))

        self.src = []
        if self.src_on :
            if isinstance(self.src_B0, (float, int)):
                self.src.append(Pulse(self.src_origins, self.src_S0, self.src_B0, self.src_evolutions))
            else:
                for o, s, b, e in zip(self.src_origins, self.src_S0, self.src_B0, self.src_evolutions):
                    self.src.append(Monopole(o, s, b, e))

    def _flw(self):
        """ Get flow parameters. """
        FLW = self._cfg['flow']
        U0 = tuple([0 for i in range(len(self.shape))])
        self.flw_type = FLW.get('type', 'None').lower()
        self.flw_components = FLW.gettuple_float('components', U0)

        if self.flw_type not in ['mean flow', ]:
            self.flw_type = None
            self.flw_components = U0

        if len(self.flw_components) != len(self.shape):
            raise ValueError(f'Mean flow component must be {len(self.shape)}d')

    def _sol(self):
        """ Get solver. """
        SOL = self._cfg['solver']
        self.resume = SOL.getboolean('resume', False)
        self.nt = SOL.getint('nt', 500)
        self.ns = SOL.getint('ns', 10)
        self.CFL = SOL.getfloat('cfl', 0.5)
        self.prb = ast.literal_eval(SOL.get('probes', '()'))
        self.save_fld = SOL.getboolean('save fields', True)
        self.vsc = SOL.getboolean('viscous fluxes', True)
        self.vrt = SOL.getboolean('vorticity', True)
        self.cpt = SOL.getboolean('shock capture', True)
        self.flt = SOL.getboolean('selective filter', True)
        self.flt_xnu_n = SOL.getfloat('selective filter n-strength', 0.2)
        self.flt_xnu_0 = SOL.getfloat('selective filter 0-strength', 0.01)
        self.it = 0

        # Check probes
        if self.prb:
            for c in self.prb:
                if any(not 0 <= c[i] < self.shape[i] for i in range(len(self.shape))):
                    raise ValueError('probes must be in the domain')

        if self.nt % self.ns:
            self.nt -= self.nt % self.ns

        if any(xnu < 0 or xnu > 1 for xnu in [self.flt_xnu_n, self.flt_xnu_0]):
            raise ValueError('Filter strength must be between 0 and 1')

    def _figs(self):
        """ Get figure parameters. """
        FIGS = self._cfg['figures']
        self.show_fig = FIGS.getboolean('show figures', True) and not self.quiet
        self.show_prb = FIGS.getboolean('show probes', True)
        self.show_bz = FIGS.getboolean('show bz', True)
        self.show_bc = FIGS.getboolean('show bc', True)
        self.fps = FIGS.getint('fps', 24)

    def _3d_to_2d(self):

        if len(self.flat) != 2:
            raise ValueError('flat must be a 2 elements tuple : (axis, location)')

        flat_ax, flat_idx = self.flat
        if flat_ax not in range(3):
            raise ValueError('flat[0] (axis) must be 0, 1, or 2')
        if flat_idx not in range(self.shape[flat_ax]):
            raise ValueError('flat[1] (index) must be in the domain')

        self.shape = tuple(s for i, s in enumerate(self.shape) if i != flat_ax)
        self.steps = tuple(s for i, s in enumerate(self.steps) if i != flat_ax)
        self.flw_components = tuple(s for i, s in enumerate(self.flw_components) if i != flat_ax)
        self.bc = ''.join(bc for i, bc in enumerate(self.bc) if i
                            not in [2*flat_ax, 2*flat_ax + 1])
        if self.origin:
            self.origin = tuple(s for i, s in enumerate(self.origin) if i != flat_ax)

        if self.obstacles:
            self.obstacles = [obs.flatten(flat_ax) for obs in self.obstacles
                              if flat_idx in obs.rn[flat_ax] and obs.ndim == 3]

        if self.ics:
            for s in self.ics:
                s.origin = tuple(s for i, s in enumerate(s.origin) if i != flat_ax)

        if self.src:
            for s in self.src:
                s.origin = tuple(s for i, s in enumerate(s.origin) if i != flat_ax)

        if self.prb:
            self.prb = [[c for i, c in enumerate(prb) if i != flat_ax] for prb in self.prb]

    def get_mesh_config(self):
        """ Get Mesh configuration. """
        args = self.shape, self.steps
        kwargs = {'origin': self.origin,
                  'bc': self.bc,
                  'obstacles': self.obstacles,
                  'bz_n': self.bz_n,
                  'bz_stretch_factor': self.bz_stretch_factor,
                  'bz_stretch_order': self.bz_stretch_order,
                  'stencil': self.stencil,
                  'free': self.free}
        if self.curvilinear_func:
            kwargs['curvilinear_func'] = self.curvilinear_func
        return args, kwargs

    def __str__(self):
        s = 'Configuration:\n\n'

        # Solver
        s += "\t* Solver : \n"
        s += f"\t\t- Viscous fluxes      : {self.vsc}\n"
        s += f"\t\t- Selective filter    : {self.flt} [nu_0={self.flt_xnu_0}, nu_n={self.flt_xnu_n}]\n"
        s += f"\t\t- Shock capture       : {self.cpt}\n\n"

        # Thermophysics
        s += "\t* Thermophysic : \n"
        s += f"\t\t- c0={self.c0:.2f} m/s, rho0={self.rho0:.2f} kg/m3, p0={self.p0:.2f} Pa\n "
        s += f'\t\t- T0={self.T0:.2f} K, nu={self.nu:.3e} m2/s\n\n'

        # Grid
        s += f"\t* Geometry : \n"
        s += f"\t\t- Grid                : {'x'.join(str(n) for n in self.shape)} points grid\n"
        s += f'\t\t- boundary conditions : {self.bc}\n'
        s += f"\t\t- Spatial step        : ({', '.join(str(n) for n in self.steps)})\n"
        if self.obstacles:
            s += f"\t\t- Obstacles           : {self.obstacles}\n"
        if self.curvilinear_func:
            s += f"\t\t- Curvilinear         : {self.curvilinear_func}\n"

        s += "\n"

        # Time
        s += f"\t* Time :\n"
        s += f"\t\t- Physical time           : {self.dt*self.nt:.5e} s.\n"
        s += f"\t\t- Time step               : dt={self.dt:.5e} s and nt={self.nt}.\n\n"

        # Sources
        if self.obstacles:
            wall_source = any('V' in o.bc for o in self.obstacles)
        else:
            wall_source = False

        if self.src or self.ics or wall_source:
            s += f"\t* Sources :\n"

        if self.ics:
            for ic in self.ics:
                s += f"\t\t- {ic}.\n"

        if self.src:
            for source in self.src:
                s += f"\t\t- {source}.\n"

        if wall_source:
            s += f"\t\t- Wall source setup"

        if self.flw_type not in self._NONE:
            s += f"\t* flow         : {self.flw_type} {self.flw_components}.\n"

        return s

    def __repr__(self):
        return self.__str__()