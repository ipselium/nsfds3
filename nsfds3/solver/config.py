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

    cfg.add_section('thermophysic')
    cfg.set('thermophysic', 'norm', 'False')
    cfg.set('thermophysic', 'p0', '101325.0')
    cfg.set('thermophysic', 't0', '20.0')
    cfg.set('thermophysic', 'gamma', '1.4')
    cfg.set('thermophysic', 'prandtl', '0.7')

    cfg.add_section('geometry')
    cfg.set('geometry', 'mesh', 'regular')
    cfg.set('geometry', 'file', 'None')
    cfg.set('geometry', 'geoname', 'helmholtz_double')
    cfg.set('geometry', 'curvname', 'None')
    cfg.set('geometry', 'Nd', '23')
    cfg.set('geometry', 'Rx', '3.')
    cfg.set('geometry', 'only_pml', 'False')
    cfg.set('geometry', 'bc', 'WWWWWW')
    cfg.set('geometry', 'nx', '256')
    cfg.set('geometry', 'ny', '256')
    cfg.set('geometry', 'nz', '256')
    cfg.set('geometry', 'ix0', '0')
    cfg.set('geometry', 'iy0', '0')
    cfg.set('geometry', 'iz0', '0')
    cfg.set('geometry', 'dx', '1')
    cfg.set('geometry', 'dy', '1')
    cfg.set('geometry', 'dz', '1')
    cfg.set('geometry', 'flat', '[]')

    cfg.add_section('BZ')
    cfg.set('BZ', 'grid points', '20')
    cfg.set('BZ', 'filter order', '3.')
    cfg.set('BZ', 'stretch level', '2.')
    cfg.set('BZ', 'stretch order', '3.')

    cfg.add_section('source')
    cfg.set('source', 'type', 'pulse')
    cfg.set('source', 'ixS', '32')
    cfg.set('source', 'iyS', '32')
    cfg.set('source', 'izS', '128')
    cfg.set('source', 's0', '1e6')
    cfg.set('source', 'b0', '5')
    cfg.set('source', 'f0', '20000')
    cfg.set('source', 'seed', 'None')
    cfg.set('source', 'wavfile', 'None')

    cfg.add_section('flow')
    cfg.set('flow', 'type', 'None')
    cfg.set('flow', 'u0', '0')
    cfg.set('flow', 'v0', '0')
    cfg.set('flow', 'w0', '0')

    cfg.add_section('eulerian fluxes')
    cfg.set('eulerian fluxes', 'stencil', '11')

    cfg.add_section('filtering')
    cfg.set('filtering', 'filter', 'True')
    cfg.set('filtering', 'stencil', '11')
    cfg.set('filtering', 'strength', '0.6')
    cfg.set('filtering', 'strength_on_walls', '0.01')

    cfg.add_section('viscous fluxes')
    cfg.set('viscous fluxes', 'viscosity', 'True')
    cfg.set('viscous fluxes', 'stencil', '7')

    cfg.add_section('shock capture')
    cfg.set('shock capture', 'shock capture', 'True')
    cfg.set('shock capture', 'stencil', '7')
    cfg.set('shock capture', 'method', 'pressure')

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
    cfg.set('save', 'resume', 'False')
    cfg.set('save', 'fields', 'True')
    cfg.set('save', 'vorticity', 'False')
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

    def __init__(self, args=None):

        # Minimal version of the config file
        self.base_version = '0.1.0'

        # Command line arguments + home
        self.args = args
        self.home = _pathlib.Path.home()
        self.path_default = self.home / '.nsfds3'
        self.cfgfile_default = self.path_default / 'nsfds3.conf'

        # Create config parser
        self.cfg = _configparser.ConfigParser(allow_no_value=True)

        # Load config file
        if isinstance(self.args, str):
            self.cfgfile = _pathlib.Path(self.args)
        else:
            self.cfgfile = getattr(self.args, 'cfgfile', None)

        # Check cfg file
        if self.cfgfile:
            self.cfgfile = _pathlib.Path(self.cfgfile)
            self.path = self.cfgfile.absolute().parent
        else:
            self.path = self.path_default
            self.cfgfile = self.cfgfile_default
            self.mkdir(self.path)     # Check if cfg dir exists. If not create it.
            self.mkcfg()           # Check if cfg file exist. If not create it

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
        CFG = cfg['configuration']
        version = CFG.get('version')
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
            self._eul()
            self._flt()
            self._vsc()
            self._cpt()
            self._sim()
            self._thp()
            self._geo()
            self._bz()
            self._src()
            self._flw()
            self._save()
            self._figs()
        except _configparser.Error as err:
            print('Bad cfg file : ', err)
            _sys.exit(1)

        du = min(self.dx, self.dy, self.dz)
        c = self.c0 + max(abs(self.U0), abs(self.V0), abs(self.W0))
        self.dt = du * self.CFL / c

    def _cfg(self):
        """ Get global parameters. """
        CFG = self.cfg['configuration']
        self.timings = getattr(self.args, 'timings', CFG.getboolean('timings', False))
        self.quiet = getattr(self.args, 'quiet', CFG.getboolean('quiet', False))
        self.cpu = CFG.getint('cpu', 1)

    def _sim(self):
        """ Get simulation parameters. """
        SIM = self.cfg['simulation']
        self.nt = getattr(self.args, 'nt', SIM.getint('nt', 500))
        self.ns = SIM.getint('ns', 10)
        self.CFL = SIM.getfloat('cfl', 0.5)

        if self.nt % self.ns:
            self.nt -= self.nt % self.ns

        self.it = 0

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
        self.nx = GEO.getint('nx', 256)
        self.ny = GEO.getint('ny', 256)
        self.nz = GEO.getint('nz', 256)
        self.ix0 = GEO.getint('ix0', 0)
        self.iy0 = GEO.getint('iy0', 0)
        self.iz0 = GEO.getint('iz0', 0)
        self.dx = GEO.getfloat('dx', 1)
        self.dy = GEO.getfloat('dy', 1)
        self.dz = GEO.getfloat('dz', 1)
        self.volumic = GEO.getboolean('volumic', True)
        self.flat = _json.loads(GEO.get('flat', '[]'))
        if self.flat and self.volumic:
            self.volumic = False
            self.flat_ax, self.flat_idx = self.flat
            self.shape = tuple(s for i, s in
                               enumerate((self.nx, self.ny, self.nz))
                               if i != self.flat_ax)
            self.origin = tuple(s for i, s in
                                enumerate((self.ix0, self.iy0, self.iz0))
                                if i != self.flat_ax)
            self.steps = tuple(s for i, s in
                               enumerate((self.dx, self.dy, self.dz))
                               if i != self.flat_ax)
            if len(self.bc) == 6:
                self.bc = ''.join(bc for i, bc in enumerate(self.bc) if i
                                  not in [2*self.flat_ax, 2*self.flat_ax + 1])
        elif not self.volumic:
            self.flat = None
            self.shape = (self.nx, self.ny)
            self.origin = (self.ix0, self.iy0)
            self.steps = (self.dx, self.dy)
            if len(self.bc) == 6:
                self.bc = self.bc[:4]
        elif not self.flat and self.volumic:
            self.shape = (self.nx, self.ny, self.nz)
            self.origin = (self.ix0, self.iy0, self.iz0)
            self.steps = (self.dx, self.dy, self.dz)

        # Mesh type and geometry
        self.mesh = GEO.get('mesh', 'regular').lower()
        self.geofile = GEO.get('file', None)
        self.geoname = GEO.get('geoname', 'square')

        if self.geofile not in self.none:
            self.geofile = self.path / self.geofile

        self.obstacles = files.get_obstacle(self)

        if self.mesh not in ['regular', 'adaptative', 'curvilinear']:
            raise ValueError('mesh must be regular, adaptative, or curvilinear')


        # Curvilinear mesh
        if self.mesh == 'curvilinear':
            self.curvname = GEO.get('curvname', None)
        else:
            self.curvname = None

        # Adaptative mesh
        if self.mesh == 'adaptative':
            self.Nd = GEO.getint('Nd', 23)                      # Adapt over Nd pts
            self.Rx = GEO.getfloat('Rx', 3.)                    # dilatation rate
            self.only_pml = GEO.getboolean('only_pml', False)   # adapt only in PML
        else:
            self.Nd, self.Rx, self.only_pml = None, None, None

    def _bz(self):
        """ Get Buffer Zone parameters. """
        BZ = self.cfg['BZ']
        self.nbz = BZ.getint('grid points', 20)
        self.forder = BZ.getfloat('filter ordrer', 3.)
        self.slevel = BZ.getfloat('stretch level', 2.)
        self.sorder = BZ.getfloat('stretch order', 3.)

    def _src(self):
        """ Get source parameters. """
        SRC = self.cfg['source']
        self.stype = SRC.get('type', 'pulse').lower()
        self.ixS = SRC.getint('ixS', 32)
        self.iyS = SRC.getint('iyS', 32)
        self.izS = SRC.getint('izS', 32)
        self.S0 = SRC.getfloat('s0', 1e3)
        self.B0 = SRC.getfloat('b0', 5)
        self.f0 = SRC.getfloat('f0', 20000)
        self.wavfile = SRC.get('wavfile', None)
        self.seed = SRC.get('seed', None)
        self.off = SRC.getint('off', self.nt)

        if self.wavfile:
            self.wavfile = _pathlib.Path(self.wavfile).expanduser()

        if self.seed not in self.none:
            try:
                self.seed = int(self.seed)
            except ValueError:
                raise ValueError('Seed must be int or None')

        if self.stype not in ['pulse', 'harmonic', 'wav', 'white']:
            self.S0 = 0

    def _flw(self):
        """ Get flow parameters. """
        FLW = self.cfg['flow']
        self.ftype = FLW.get('type', 'None').lower()
        self.U0 = FLW.getfloat('u0', 0)
        self.V0 = FLW.getfloat('v0', 0)
        self.V0 = FLW.getfloat('w0', 0)

        if self.ftype not in ['custom', 'vortex', 'poiseuille']:
            self.ftype = None
            self.U0, self.V0, self.W0 = 0., 0., 0.

    def _eul(self):
        """ Get Euler fluxes parameters. """
        EUL = self.cfg['eulerian fluxes']
        self.stencil = EUL.getint('stencil', 11)

        if self.stencil not in [3, 7, 11]:
            raise ValueError('stencil must be 3, 7 or 11')

    def _flt(self):
        """ Get selective filter parameters. """
        FLT = self.cfg['filtering']
        self.flt = FLT.getboolean('filter', True)
        self.flt_stencil = FLT.getint('stencil', 11)
        self.flt_xnu = FLT.getfloat('strength', 0.2)
        self.flt_xnu0 = FLT.getfloat('strength_on_walls', 0.01)

        if self.flt_stencil not in [7, 11]:
            raise ValueError('only 7 and 11 pts filters implemented for now')

        if any(xnu < 0 or xnu > 1 for xnu in [self.flt_xnu, self.flt_xnu0]):
            raise ValueError('Filter strength must be between O and 1')

    def _vsc(self):
        """ Get viscous fluxes parameters. """
        VSC = self.cfg['viscous fluxes']
        self.vsc = VSC.getboolean('viscosity', True)
        self.vsc_stencil = VSC.getint('stencil', 3)

        if self.vsc_stencil not in [3, 7, 11]:
            raise ValueError('viscous fluxes only available with 3, 7 or 11 pts')

    def _cpt(self):
        """ Get shock capture parameters. """
        CPT = self.cfg['shock capture']
        self.cpt = CPT.getboolean('shock capture', True)
        self.cpt_stencil = CPT.getint('stencil', 7)
        self.cpt_meth = CPT.get('method', 'pressure').lower()
        self.cpt_rth = 1e-6

        if self.cpt_stencil not in [3, 7, 11]:
            raise ValueError('capture only available with 3, 7 or 11 pts')

        if self.cpt_meth not in ['pressure', 'dilatation']:
            raise ValueError('capture method must be pressure or dilatation')

    def _save(self):
        """ Get save parameters. """
        SAVE = self.cfg['save']
        if self.path == self.path_default:
            self.savepath = _pathlib.Path(SAVE.get('path', 'nsfds3/'))
        else:
            self.savepath = self.path / SAVE.get('path', 'nsfds3/')
        self.savefile = SAVE.get('filename', 'tmp') + '.hdf5'
        self.comp = SAVE.get('compression', 'None')
        self.comp = None if self.comp == 'None' else self.comp

        self.resume = SAVE.getboolean('resume', False)
        self.save_fields = SAVE.getboolean('fields', True)
        self.save_vortis = SAVE.getboolean('vorticity', False)
        self.probes = _json.loads(SAVE.get('probes', '[]'))

        # Check probes
        if self.probes:
            for c in self.probes:
                if not 0 <= c[0] < self.nx \
                   or not 0 <= c[1] < self.ny \
                   or not 0 <= c[2] < self.nz:
                    raise ValueError('probes must be in the domain')

        # if self.savepath does not exist, create it
        self.mkdir(self.savepath)

        self.datafile = getattr(self.args, 'datafile', None)
        if not self.datafile:
            self.datafile = self.savepath / self.savefile
        else:
            self.datafile = _pathlib.Path(self.datafile).expanduser()

    def _figs(self):
        """ Get figure parameters. """
        FIGS = self.cfg['figures']
        self.figures = FIGS.getboolean('figures', True) and not self.quiet
        self.show_probes = FIGS.getboolean('show_probes', True)
        self.show_bz = FIGS.getboolean('show_bz', True)
        self.show_bc_profiles = FIGS.getboolean('show_bc_profiles', True)
        self.fps = FIGS.getint('fps', 24)
        self.xlim = _json.loads(FIGS.get('xlim', '[]'))
        self.ylim = _json.loads(FIGS.get('ylim', '[]'))
        self.zlim = _json.loads(FIGS.get('zlim', '[]'))

    def get_config(self):
        """ Get configuration. """
        return (self.shape, self.steps), \
                {'origin': self.origin,
                 'bc': self.bc,
                 'obstacles': self.obstacles,
                 'nbz': self.nbz,
                 'slevel': self.slevel,
                 'sorder': self.sorder,
                 'stencil': self.stencil,
                 'flat': self.flat}
