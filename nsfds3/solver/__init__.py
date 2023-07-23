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
# Creation Date : 2022-07-08 - 13:27:17
"""
The `solver` package contains the following main objects:

* :py:class:`CfgSetup`: read a configuration file and set all simulation parameters.
* :py:class:`FDTD`: setup and run the FDTD simulation
* :py:class:`Pulse`: Describes an initial gaussian pressure pulse condition
* :py:class:`Monopole`: Describes a time evolving source


 The following example gives the general philosophy to use **nsfds3**::

   from nsfds3.solver import CfgSetup, FDTD
   from nsfds3.cpgrid import build_mesh

   # Initialize simulation parameter
   cfg = CfgSetup()    # or cfg = CfgSetup('path_to_configfile.conf')

   # Define the mesh
   msh = build_mesh(cfg)

   # Create and run simulation
   fdtd = FDTD(msh, cfg)
   fdtd.run()

   # Figures
   fdtd.show(view='p', nans=True, buffer=True, grid=False, obstacles=True)
"""

from nsfds3.solver.config import CfgSetup
from nsfds3.solver.fdtd import FDTD
from nsfds3.solver.sources import Pulse, Monopole

__all__ = ['CfgSetup', 'FDTD', 'Pulse', 'Monopole']
