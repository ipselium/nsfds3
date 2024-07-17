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

import time
import dataclasses
import numpy as np


@dataclasses.dataclass
class Timing:

    name: str
    mean: float
    std: float
    iterations: int


def get(lst, attr):
    return [getattr(item, attr) for item in lst]


def ratios(timings):
    cython_funcs = [item for item in timings if not item.name.startswith('c')]
    c_funcs = [item for item in timings if item.name.startswith('c')]

    names = [item.name for item in cython_funcs]
    ratio = [i.mean / j.mean for i, j in zip(cython_funcs, c_funcs)]
    return names, ratio


def time_it(funcs, args=None, tmax=1., Nmin=100000):

    if not isinstance(funcs, (list, tuple)):
        funcs = (funcs, )

    if not args:
        args = dict()

    out = []

    for f in funcs:
        times = []
        n = 0
        tg = time.perf_counter()
        while True:
            ti = time.perf_counter()
            f(*args)
            tf = time.perf_counter() - ti
            times.append(tf)
            n += 1
            if n > Nmin or time.perf_counter() - tg > tmax:
                break

        times = np.array(times)
        out.append(Timing(name=f.__name__, mean=times.mean(), std=times.std(), iterations=n-1))

    return out


def compare(fld, funcs, args):

    for i in range(0, len(funcs), 2):

        if 'x' in funcs[i].__name__:
            dv = fld.ru
        elif 'y' in funcs[i].__name__:
            dv = fld.rv
        else:
            dv = fld.rw

        dv[...] = 0
        funcs[i](*args)
        K1 = np.array(dv).copy()

        dv[...] = 0
        funcs[i+1](*args)
        K2 = np.array(dv).copy()

        if not np.all(K1 == K2) or not np.allclose(K1, K2):
            print(f'\n{funcs[i].__name__} & {funcs[i+1].__name__} : ', np.all(K1 == K2), np.allclose(K1, K2))