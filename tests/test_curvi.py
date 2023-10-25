#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2023 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of {name}
#
# {name} is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# {name} is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with {name}. If not, see <http://www.gnu.org/licenses/>.
#
# Creation Date : 2023-03-31 - 15:50:27
"""
-----------
DOCSTRING

-----------
"""

import numpy as np
import matplotlib.pyplot as plt

from nsfds3.cpgrid import CartesianGrid, CurvilinearGrid
from nsfds3.solver import CfgSetup, FDTD

import faulthandler


def func(x, y):
    import numpy as np
    xsine = np.linspace(-np.pi, np.pi, x.shape[0])
    sine = np.sin(xsine/0.1)
    profile = np.zeros_like(x)
    for i in range(x.shape[1]):
        profile[:, i] = (2/(i/50+1)) * (sine - xsine**2)

    return x.copy(), y + profile



#def func(x, y):
#    return x, y

# Initialize configuration & mesh (overlapped2)
cfgfile = '/home/cdesjouy/.nsfds3/nsfds3_true2d.conf'
cfgfile = '/home/cdesjouy/.nsfds3/nsfds3.conf'
#cfgfile = '/home/cdesjouy/.nsfds3/flow.conf'

cfg = CfgSetup(cfgfile)
args, kwargs = cfg.get_mesh_config()
kwargs['bc'] = 'PPWW'
msh = CurvilinearGrid(*args, curvilinear_func=func, **kwargs)
#print(msh)
#print(msh.domains)

# Initialize solver
cfg.nt = 1
cfg.vsc = False
cfg.flt = False
cfg.cpt = False
cfg.S0 = 1e1

fdtd = FDTD(cfg, msh, timings=False)

faulthandler.enable()
fdtd.run()
