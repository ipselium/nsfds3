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
# Creation Date : 2022-07-11 - 22:25:34
"""
-----------
DOCSTRING

-----------
"""


import itertools as _it
from time import perf_counter as _pc
import numpy as np
import matplotlib.pyplot as plt

from libfds.fields import Fields2d
from libfds.fluxes import EulerianFluxes2d
from libfds.filters import SelectiveFilter, ShockCapture
from mplutils.custom_cmap import modified_jet, MidPointNorm

from rich import print
from rich.panel import Panel
from rich.progress import track

from nsfds3.cpgrid import RegularMesh
from nsfds3.solver import CfgSetup, FDTD


if __name__ == '__main__':
    config = CfgSetup()
    args, kwargs = config.get_config()
    mesh = RegularMesh(*args, **kwargs)
    fdtd = FDTD(config, mesh)
    print(mesh)
    print(mesh.domains)
    fdtd.run()
    fdtd.show()
