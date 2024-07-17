#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2024 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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

from nsfds3.solver import CfgSetup, CustomInitialConditions
from nsfds3.cpgrid import build_mesh
from libfds.cfdtd import cFdtd3d

Nx, Ny, Nz = 1000, 1000, 32

cfg = CfgSetup()
cfg.geo.steps = (1e-2, 1e-2, 1e-2)
cfg.geo.shape = (Nx, Ny, Nz)
msh = build_mesh(cfg)
ci = CustomInitialConditions(cfg, msh)
fld = Fields3d(cfg, msh, ci)