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
# Creation Date : 2023-07-10 - 14:56:26
"""
-----------

Sources
-----------
"""

import numpy as np


class Pulse:

    def __init__(self, origin, amplitude=1, width=5):
        self.origin = origin
        self.amplitude = amplitude
        self.width = width

    def __str__(self):
        return f'{type(self).__name__} @ {self.origin} [S0={self.amplitude}/B0={self.width}]'

    def __repr__(self):
        return self.__str__()


class Monopole(Pulse):

    def __init__(self, origin, amplitude=1, width=5, evolution=None):
        super().__init__(origin=origin, amplitude=amplitude, width=width)
        self.evolution = None
        self._f = evolution

    def set_evolution(self, nt, dt):
        """ Set time evolution of the source.

            Parameters
            ----------
            nt : int
                Number of time steps
            dt : float
                Time step
        """
        if isinstance(self._f, (int, float)):
            t = np.linspace(0, nt * dt, nt + 1)
            f = self._f
            self.evolution = self.amplitude * np.sin(2 * np.pi * f * t)

        elif callable(self._f):
            self.evolution = self.amplitude * self.evolution(nt, dt)
