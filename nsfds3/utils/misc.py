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

Some tools used by the mesher.

-----------
"""

import sys as _sys
import datetime as _datetime


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
