#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2020 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
# Creation Date : 2022-06-28 - 16:05:58
"""
-----------
DOCSTRING

-----------
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mesh import RegularMesh
from config import CfgSetup

from ofdlib3.fields import Fields2d
from ofdlib3.efluxes import EulerianFluxes2d
from ofdlib3.sfilter import SelectiveFilter
from ofdlib3.scapture import ShockCapture


cfg = CfgSetup()
args, kwargs = cfg.get_config()
msh = RegularMesh(*args, **kwargs)

# cython
fld = Fields2d(cfg, msh)
efluxes = EulerianFluxes2d(fld)

N = 1000

# RK4 execution time
ti = time.perf_counter()
for _ in range(N):
    efluxes.rk4()
print(f'efluxes = {(time.perf_counter() - ti)/N} s.')


# Filter execuration time
fld = Fields2d(cfg, msh)
efluxes = EulerianFluxes2d(fld)
sfilter = SelectiveFilter(fld)

ti = time.perf_counter()
for _ in range(N):
    sfilter.apply()
print(f'filter = {(time.perf_counter() - ti)/N} s.')

# Capture execuration time
fld = Fields2d(cfg, msh)
efluxes = EulerianFluxes2d(fld)
sfilter = SelectiveFilter(fld)
capture = ShockCapture(fld)

ti = time.perf_counter()
for _ in range(N):
    capture.apply()
print(f'filter = {(time.perf_counter() - ti)/N} s.')


# Results
fld = Fields2d(cfg, msh)
efluxes = EulerianFluxes2d(fld)
sfilter = SelectiveFilter(fld)
capture = ShockCapture(fld)


fig, axes = plt.subplots(1, 2, figsize=(9, 4))


p = np.array(fld.p) - cfg.p0
axes[0].imshow(p, vmin=0, vmax=cfg.S0 / 10)

for i in range(500):
    efluxes.rk4()
    sfilter.apply()
    capture.apply()

p = np.array(fld.p) - cfg.p0
axes[1].imshow(p, vmin=0, vmax=cfg.S0 / 10)

print(fld.residual())

plt.show()
