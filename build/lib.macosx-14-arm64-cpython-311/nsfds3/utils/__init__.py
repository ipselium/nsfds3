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
# Creation Date : 2022-07-08 - 13:27:28
"""
The `utils` package gathers some utilities for **nsfds3**. The main ones are :

    * :py:func:`get_objects`: get cfg and msh from pickles
    * :py:func:`get_pressure`: get pressure from conservative variables
    * :py:func:`probes_to_wavfile`: Make .wav files from probes signals.
"""

from nsfds3.utils.files import get_objects, probes_to_wavfile
from nsfds3.utils.data import get_pressure

__all__ = ['get_objects', 'get_pressure', 'probes_to_wavfile']
