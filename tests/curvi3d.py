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



from nsfds3.cpgrid import build_mesh
from nsfds3.solver import CfgSetup, FDTD

# Initialize configuration & mesh
# 600 Mo de ram pour cfg.shape = 140, 128, 96 = 
# 13 Go de ram pour cfg.shape = 800, 800, 100

cfgfile = 'configurations/curvi_3d.conf'

cfg = CfgSetup(cfgfile)
msh = build_mesh(cfg)

fdtd = FDTD(cfg, msh, timings=False)
fdtd.run()