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

import json as _json
import time as _time
import sys as _sys
import shutil as _shutil
import datetime as _datetime
import pathlib as _pathlib
import configparser as _configparser
from pkg_resources import parse_version as _parse_version
from nsfds3.utils import files


def _parse_int_tuple(input):
    if input.lower() not in [None, 'none']:
        return tuple(int(k.strip()) for k in input[1:-1].split(','))
    return None


def _parse_float_tuple(input):
    if input.lower() not in [None, 'none']:
        return tuple(float(k.strip()) for k in input[1:-1].split(','))
    return None


def create_template(path=None, filename=None, cfg=None):
    """ Create default configuration file. Default location is .nsfds3/nsfds3.conf."""

    if not cfg:
        cfg = _configparser.ConfigParser(allow_no_value=True)

    cfg.add_section('configuration')
    cfg.set('configuration', 'version', '0.1.0') #str(nsfds3.__version__))
    cfg.set('configuration', 'timings', 'False')
    cfg.set('configuration', 'quiet', 'False')
    cfg.set('configuration', 'cpu', '1')

    cfg.add_section('simulation')
    cfg.set('simulation', 'nt', '500')
    cfg.set('simulation', 'ns', '10')
    cfg.set('simulation', 'cfl', '0.5')
    cfg.set('simulation', 'resume', 'False')

    cfg.add_section('thermophysic')
    cfg.set('thermophysic', 'norm', 'False')
    cfg.set('thermophysic', 'p0', '101325.0')
    cfg.set('thermophysic', 't0', '20.0')
    cfg.set('thermophysic', 'gamma', '1.4')
    cfg.set('thermophysic', 'prandtl', '0.7')

    cfg.add_section('geometry')
    cfg.set('geometry', 'mesh', 'cartesian')
    cfg.set('geometry', 'file', 'None')
    cfg.set('geometry', 'geoname', 'single')
    cfg.set('geometry', 'curvname', 'None')
    cfg.set('geometry', 'bc', 'WWWWWW')
    cfg.set('geometry', 'shape', '(256, 256, 256)')
    cfg.set('geometry', 'origin', 'None')
    cfg.set('geometry', 'steps', '(1., 1., 1.)')
    cfg.set('geometry', 'flat', '[]')

    cfg.add_section('buffer zone')
    cfg.set('buffer zone', 'grid points', '20')
    cfg.set('buffer zone', 'filter order', '3.')
    cfg.set('buffer zone', 'stretch level', '2.')
    cfg.set('buffer zone', 'stretch order', '3.')

    cfg.add_section('acoustic source')
    cfg.set('source', 'type', 'pulse')
    cfg.set('source', 'origin', '(32, 32, 32)')
    cfg.set('source', 's0', '1e6')
    cfg.set('source', 'b0', '5')

    cfg.add_section('flow')
    cfg.set('flow', 'type', 'None')
    cfg.set('flow', 'components', '(0, 0, 0)')

    cfg.add_section('solver')
    cfg.set('solver', 'viscous fluxes', 'True')
    cfg.set('solver', 'vorticity', 'True')
    cfg.set('solver', 'shock capture', 'True')
    cfg.set('solver', 'selective filter', 'True')
    cfg.set('solver', 'selective filter n-strength', '0.6')
    cfg.set('solver', 'selective filter 0-strength ', '0.01')

    cfg.add_section('figures')
    cfg.set('figures', 'figures', 'True')
    cfg.set('figures', 'show_probes', 'True')
    cfg.set('figures', 'show_bz', 'True')
    cfg.set('figures', 'show_bc_profiles', 'True')
    cfg.set('figures', 'fps', '24')

    cfg.add_section('save')
    cfg.set('save', 'path', 'nsfds3/')
    cfg.set('save', 'filename', 'tmp')
    cfg.set('save', 'compression', 'None')
    cfg.set('save', 'probes', '[]')

    if not path:
        path = _pathlib.Path.home() / '.nsfds3'

    if not filename:
        filename = 'nsfds3.conf'

    if not path.is_dir():
        path.mkdir()

    with open(path / filename, 'w') as cf:
        cfg.write(cf)


class CfgSetup:
    """ Handle configuration file. """

    def __init__(self, cfgfile=None):

        # Minimal version of the config file
        self.base_version = '0.1.0'

        # Default configuration
        self.home = _pathlib.Path.home()
        self.path_default = self.home / '.nsfds3'
        self.cfgfile_default = self.path_default / 'nsfds3.conf'

        # Create config parser
        self.cfg = _configparser.ConfigParser(allow_no_value=True,
                                              converters={'tuple_int': _parse_int_tuple,
                                                          'tuple_float': _parse_float_tuple})

        # Load config file
        if isinstance(cfgfile, (_pathlib.Path, str)):
            self.cfgfile = _pathlib.Path(cfgfile)
            self.path = self.cfgfile.absolute().parent
        else:
            self.path = self.path_default
            self.cfgfile = self.cfgfile_default
            self.mkdir(self.path)     # Check if cfg dir exists. If not create it.
            self.mkcfg()              # Check if cfg file exist. If not create it

        # Check if config file version is ok
        self.check_version()

        # Read config file (can be overridden by command line)
        self.cfg.read(self.cfgfile)

        # Parse arguments
        self.run()

    @staticmethod
    def mkdir(directory):
        """ Check if dir exists. If not, create it."""
        if not directory.is_dir():
            directory.mkdir()
            print("Create directory :", directory)
            _time.sleep(0.5)

    def mkcfg(self):
        """ Check if config file exists. If not create it. """
        if not (self.path / 'nsfds3.conf').is_file():
            open(self.path / 'nsfds3.conf', 'a').close()
            print(f"Create configuration file : {self.path}/nsfds3.conf")
            _time.sleep(0.5)
            create_template(cfg=self.cfg)

    def check_version(self):
        """ Check version of the config file. Overwrite it if too old. """
        cfg = _configparser.ConfigParser(allow_no_value=True)
        cfg.read(self.cfgfile)
        version = cfg['configuration'].get('version')
        is_default_cfg = self.cfgfile == self.cfgfile_default
        is_version_ok = _parse_version(version) >= _parse_version(self.base_version)

        if not is_version_ok and is_default_cfg:
            self._overwrite_config_file()
        elif not is_version_ok:
            print(f'Config file version must be >= {self.base_version}')
            _sys.exit(1)

    def _overwrite_config_file(self):
        """ Create new default config file. """
        # Backup old config file
        now = _datetime.datetime.now()
        name = f'{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}'
        _shutil.move(self.path / 'nsfds3.conf', self.path / f'nsfds3_{name}.conf')

        print(f'Current configfile backup : nsfds3_{name}.conf')
        _time.sleep(1)

        # Create new config file
        self.mkcfg()

    def run(self):
        """ Run configuration. """

        self.none = ['', 'none', 'None', None]

        try:
            self._cfg()
            self._sol()
            self._sim()
            self._thp()
            self._geo()
            self._bz()
            self._src()
            self._flw()
            self._save()
            self._figs()
            if self.flat and len(self.shape) == 3:
                self._3d_to_2d()
        except _configparser.Error as err:
            print('Bad cfg file : ', err)
            _sys.exit(1)

        du = min(self.steps)
        c = self.c0 + max([abs(u) for u in self.U])
        self.dt = du * self.CFL / c

    def _cfg(self):
        """ Get global parameters. """
        CFG = self.cfg['configuration']
        self.timings = CFG.getboolean('timings', False)
        self.quiet = CFG.getboolean('quiet', False)
        self.cpu = CFG.getint('cpu', 1)

    def _sim(self):
        """ Get simulation parameters. """
        SIM = self.cfg['simulation']
        self.nt = SIM.getint('nt', 500)
        self.ns = SIM.getint('ns', 10)
        self.CFL = SIM.getfloat('cfl', 0.5)
        self.resume = SIM.getboolean('resume', False)
        self.it = 0

        if self.nt % self.ns:
            self.nt -= self.nt % self.ns

    def _thp(self):
        """ Get thermophysical parameters. """
        THP = self.cfg['thermophysic']
        self.norm = THP.getboolean('norm', False)
        self.Ssu = 110.4        # Sutherland constant
        self.Tref = 273.15
        self.T0 = self.Tref + THP.getfloat('t0', 20.0)
        self.prandtl = THP.getfloat('prandtl', 0.7)
        self.gamma = THP.getfloat('gamma', 1.4)
        self.cv = 717.5
        self.cp = self.cv * self.gamma
        self.p0 = THP.getfloat('p0', 101325.0)
        self.rho0 = self.p0 / (self.T0 * (self.cp - self.cv))
        self.c0 = (self.gamma * self.p0 / self.rho0)**0.5
	# Dynamic viscosity at T = 0 deg. C
        self.mu0 = 0.00001716
	# Dynamic viscosity at T0
        self.mu = (self.mu0 * (self.T0 / self.Tref)**(3. / 2.) *
                   (self.Tref + self.Ssu) / (self.T0 + self.Ssu))
	# Kinematic viscosity at T0
        self.nu = self.mu / self.rho0

        if self.c0 < 1:
            raise ValueError('c0 must be >= 1')

        if self.norm:
            self.rho0 = 1
            self.c0 = 1
            self.p0 = self.rho0 * self.c0**2 / self.gamma
            self.T0 = 299.8189

    def _geo(self):
        """ Get geometry parameters. """
        GEO = self.cfg['geometry']

        # Grid
        self.bc = GEO.get('bc', 'WWWWWW').upper()
        self.shape = GEO.gettuple_int('shape', (256, 256, 256))
        self.origin = GEO.gettuple_int('origin', None)
        self.steps = GEO.gettuple_float('steps', (1., 1., 1.))
        self.flat = GEO.gettuple_int('flat', None)

        # Mesh type and geometry
        self.mesh = GEO.get('mesh', 'cartesian').lower()
        self.geofile = GEO.get('file', None)
        self.geoname = GEO.get('geoname', 'square')

        if len(self.shape) != len(self.steps) or 2*len(self.shape) != len(self.bc):
            raise ValueError('shape, steps and bc must have coherent dims.')

        if self.mesh not in ['cartesian', 'curvilinear']:
            raise ValueError('mesh must be cartesian or curvilinear')

        # Geometry
        if self.geofile not in self.none:
            self.geofile = self.path / self.geofile
        self.obstacles = files.get_obstacle(self)

        # Curvilinear mesh
        if self.mesh == 'curvilinear':
            self.curvname = GEO.get('curvname', None)
        else:
            self.curvname = None

    def _bz(self):
        """ Get Buffer Zone parameters. """
        BZ = self.cfg['buffer zone']
        self.nbz = BZ.getint('grid points', 20)
        self.sigma_order = BZ.getfloat('filter ordrer', 3.)
        self.stretch_factor = BZ.getfloat('stretch factor', 2.)
        self.stretch_order = BZ.getfloat('stretch order', 3.)

    def _src(self):
        """ Get source parameters. """
        SRC = self.cfg['acoustic source']
        self.stype = SRC.get('type', 'pulse').lower()
        self.sorigin = SRC.gettuple_int('origin', tuple([int(n/2) for n in self.shape]))
        self.S0 = SRC.getfloat('amplitude', 1e3)
        self.B0 = SRC.getfloat('width', 5)

        if self.stype not in ['pulse', ]:
            self.stype = None
            self.S0 = 0

        if len(self.sorigin) != len(self.shape):
            raise ValueError(f'Source location must be {len(self.shape)}d')

    def _flw(self):
        """ Get flow parameters. """
        FLW = self.cfg['flow']
        U0 = tuple([0 for i in range(len(self.shape))])
        self.ftype = FLW.get('type', 'None').lower()
        self.U = FLW.gettuple_float('components', U0)

        if self.ftype not in ['mean flow', ]:
            self.ftype = None
            self.U = U0

        if len(self.U) != len(self.shape):
            raise ValueError(f'Mean flow component must be {len(self.shape)}d')

    def _sol(self):
        """ Get solver. """
        SOL = self.cfg['solver']
        self.vsc = SOL.getboolean('viscous fluxes', True)
        self.vrt = SOL.getboolean('vorticity', True)
        self.cpt = SOL.getboolean('shock capture', True)
        self.flt = SOL.getboolean('selective filter', True)
        self.flt_xnu_n = SOL.getfloat('selective filter n-strength', 0.2)
        self.flt_xnu_0 = SOL.getfloat('selective filter 0-strength', 0.01)

        if any(xnu < 0 or xnu > 1 for xnu in [self.flt_xnu_n, self.flt_xnu_0]):
            raise ValueError('Filter strength must be between O and 1')

    def _save(self):
        """ Get save parameters. """
        SAVE = self.cfg['save']
        if self.path == self.path_default:
            self.savepath = _pathlib.Path(SAVE.get('path', 'nsfds3/'))
        else:
            self.savepath = self.path / SAVE.get('path', 'nsfds3/')
        self.savefile = SAVE.get('filename', 'tmp') + '.hdf5'
        self.save_fld = SAVE.getboolean('fields', True)
        self.comp = SAVE.get('compression', 'None')
        self.comp = None if self.comp == 'None' else self.comp
        self.prb = _json.loads(SAVE.get('probes', '[]'))

        # Check probes
        if self.prb:
            for c in self.prb:
                if any(not 0 <= c[i] < self.shape[i] for i in range(len(self.shape))):
                    raise ValueError('probes must be in the domain')

        # if self.savepath does not exist, create it
        self.mkdir(self.savepath)
        self.datafile = self.savepath / self.savefile

    def _figs(self):
        """ Get figure parameters. """
        FIGS = self.cfg['figures']
        self.figures = FIGS.getboolean('figures', True) and not self.quiet
        self.show_prb = FIGS.getboolean('show_probes', True)
        self.show_bz = FIGS.getboolean('show_bz', True)
        self.show_bc = FIGS.getboolean('show_bc', True)
        self.fps = FIGS.getint('fps', 24)

    def _3d_to_2d(self):

        if len(self.flat) != 2:
            raise ValueError('flat must be a 2 elements tuple : (axis, location)')

        self.flat_ax, self.flat_idx = self.flat
        if self.flat_ax not in range(3):
            raise ValueError('flat[0] (axis) must be 0, 1, or 2')
        if self.flat_idx not in range(self.shape[self.flat_ax]):
            raise ValueError('flat[1] (index) must be in the domain')

        self.shape = tuple(s for i, s in enumerate(self.shape) if i != self.flat_ax)
        self.steps = tuple(s for i, s in enumerate(self.steps) if i != self.flat_ax)
        self.U = tuple(s for i, s in enumerate(self.U) if i != self.flat_ax)
        self.sorigin = tuple(s for i, s in enumerate(self.sorigin) if i != self.flat_ax)
        self.bc = ''.join(bc for i, bc in enumerate(self.bc) if i
                            not in [2*self.flat_ax, 2*self.flat_ax + 1])
        if self.origin:
            self.origin = tuple(s for i, s in enumerate(self.origin) if i != self.flat_ax)

        if self.obstacles:
            self.obstacles = [obs.flatten(self.flat_ax) for obs in self.obstacles
                              if self.flat_idx in obs.ranges[self.flat_ax] and obs.volumic]

        if self.prb:
            self.prb = [[c for i, c in enumerate(prb) if i != self.flat_ax] for prb in self.prb]

    def get_mesh_config(self):
        """ Get Mesh configuration. """
        return (self.shape, self.steps), \
                {'origin': self.origin,
                 'bc': self.bc,
                 'obstacles': self.obstacles,
                 'nbz': self.nbz,
                 'stretch_factor': self.stretch_factor,
                 'stretch_order': self.stretch_order}

    def __str__(self):
        s = 'Simulation parameters:'

        # Thermophysics
        s += "\t* Thermophysic : "
        s += f"c0={self.c0:.2f} m/s, rho0={self.rho0:.2f} kg/m3, p0={self.p0:.2f} Pa\n "
        s += f'\t\t\t T0={self.T0:.2f} K, nu={self.nu:.3e} m2/s\n'

        # Time
        s += f"\t* Physical time: {self.dt*self.nt:.5e} s.\n"
        s += f"\t* Time step    : dt={self.dt:.5e} s and nt={self.nt}.\n"

        # Source
        if self.stype not in self.none:
            s += f"\t* source       : {self.stype} at {self.sorigin}"

        if self.ftype not in self.none:
            s += f"\t* flow         : {self.ftype} {self.components}."

        return s

    def __repr__(self):
        return self.__str__()