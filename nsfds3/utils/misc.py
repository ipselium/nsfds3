#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2023 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
# Creation Date : 2022-05-18 - 22:03:45
# pylint: disable=too-few-public-methods
"""
-----------

Some misc. tools.

-----------
"""

import os as _os
import sys as _sys
import datetime as _datetime
import itertools as _it
import numpy as _np
import scipy.signal as _sps
import scipy.io.wavfile as _wf
from time import perf_counter as _pc
from rich.color import ANSI_COLOR_NAMES


font_colors = list(ANSI_COLOR_NAMES.keys())[30:]


def getsizeof(obj, seen=None, unit=None):
    """Recursively finds size of objects in bytes."""
    scale = 1e-3 if unit == 'k' else 1e-6 if unit == 'M' else 1e-9 if unit == 'G' else 1
    size = _sys.getsizeof(obj) * scale
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(getsizeof(v, seen=seen, unit=unit) for v in obj.values())
        size += sum(getsizeof(k, seen=seen, unit=unit) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += getsizeof(obj.__dict__, seen=seen, unit=unit)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(getsizeof(i, seen=seen, unit=unit) for i in obj)
    return size


def secs_to_dhms(secs):
    """ Convert seconds to years, months, days, hh:mm:ss."""

    dhms = _datetime.datetime(1, 1, 1) + _datetime.timedelta(seconds=secs)

    year, years = f'{dhms.year-1} year, ', f'{dhms.year-1} years, '
    month, months = f'{dhms.month-1} month, ', f'{dhms.month-1} months, '
    day, days = f'{dhms.day-1} day, ', f'{dhms.day-1} days, '
    h = f'{dhms.hour}:'
    m = f'{dhms.minute:02}:'
    s = f'{dhms.second:02}:'
    ms = f'{str(dhms.microsecond)[:2]}'

    return (year if dhms.year == 2 else years if dhms.year > 2 else '') + \
           (month if dhms.month == 2 else months if dhms.month > 2 else '') + \
           (day if dhms.day == 2 else days if dhms.day > 2 else '') + \
           (h if dhms.hour > 0 else '') + m + s + ms


def timer(func):
    """ Time method of a class instance containing:

        - _timings: dictionary attribute
        - timing: boolean attribute
    """
    def wrapper(self, *args, **kwargs):

        if self.timings:
            start = _pc()

        func(self, *args, **kwargs)

        if self.timings:
            if func.__name__ in self._timings:
                self._timings[func.__name__].append(_pc() - start)
            else:
                self._timings[func.__name__] = [_pc() - start, ]

    return wrapper


def unload_timings(timings, eps=2e-4):
    """ Empty timings and return a formatted description of the timings and the mean time per iteration.

    Parameters
    ----------
    timings: dict
        Dictionary containing the timings
    eps: float, optional
        Display only timings > eps. eps=2e-4 by default.

    Returns
    -------
    tuple with str and float : tuple
        formatted description of the timings (str) and total time (float)

    """

    desc = ""

    if timings:
        ns = len(list(timings.values())[0])
        time_per_iteration = sum(list(_it.chain(*timings.values()))) / ns
    else:
        time_per_iteration = 0

    for color, key in zip(font_colors, timings):
        title = key if not key.startswith('_') else key[1:]
        time = _np.array(timings[key]).mean()
        if time > eps:
            desc += f'\t-[italic {color}]{title:20}: '
            desc += f'{time:.4f}\n'
        timings[key] = []

    return desc, time_per_iteration


def get_padded(s, N, value=0):
    """ Pad signal with value. """
    if N > s.shape[0]:
        return _np.concatenate([s, value*_np.ones(N - s.shape[0])])

    return s


def resample(file, target_rate, pad=None, write=False, force_mono=True):
    """ Resample target wave file with target_rate. """

    target_rate = int(target_rate)

    if not 1 < target_rate < 4.3e6:
        raise ValueError('Sampling rate must be 1 < rate < 4.3e6')

    path = _os.path.dirname(file)
    filename = _os.path.basename(file).split('.')
    rate, data = _wf.read(file)
    dtype = data.dtype
    duration = data.shape[0]/rate
    N = int(target_rate*duration)

    if len(data.shape) == 2 and force_mono:   # stereo to mono
        data = (data[:, 0] + data[:, 1])/2

    print(f'Resampling {file} at {target_rate} kHz ({N} points)...')
    print(f'Set nt > {N} to play the whole sound')
    if len(data.shape) == 1:   # mono
        data_r = _sps.resample(data, N).astype(dtype)
        if pad:
            data_r = get_padded(data_r, pad)


    if len(data.shape) == 2:   # stereo
        tmp_l = _sps.resample(data[:, 0], N).astype(dtype)
        tmp_r = _sps.resample(data[:, 1], N).astype(dtype)
        if pad:
            tmp_l = get_padded(tmp_l, pad)
            tmp_r = get_padded(tmp_r, pad)

        data_r = _np.vstack([tmp_l, tmp_r]).T


    if write:
        print(f'Writing {N} samples at {target_rate} kHz rate...')
        _wf.write(path + '{}_r.{}'.format(*filename), rate=target_rate, data=data_r)

    return data_r/abs(data_r).max()