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
# Creation Date : 2023-07-10 - 14:56:26
"""
The module `sources` provides :

    * The `Pulse` class : Describes an initial pressure condition
    * The `Monopole` class : Describes a time evolving source
"""

import numpy as _np


class Pulse:
    """ Pressure Gaussian pulse as initial condition.

    Parameters
    ----------
    origin : tuple
        Initial position of the pulse.
    amplitude : float, optional
        Amplitude of the pulse in Pa.
    width : int, optional
        Width of the pulse in number of spatial steps.
    """

    def __init__(self, origin, amplitude=1, width=5):
        self.origin = origin
        self.amplitude = amplitude
        self.width = width

    def __str__(self):
        return f'{type(self).__name__} @ {self.origin} [S0={self.amplitude}/B0={self.width}]'

    def __repr__(self):
        return self.__str__()


class Monopole(Pulse):
    """ Gaussian source evolving in time.

    Parameters
    ----------
    origin : tuple
        Position of the source.
    amplitude : float, optional
        Amplitude of the source.
    width : int, optional
        Width of the source in number of spatial steps.
    evolution : float or func, optional
        Time evolution of the source.
        If evolution is a float, it will describe the frequency of a sinusoidal time evolution.
        If evolution is a function, the time evolution of the source will be the result of
        `evolution(t)` where `t` is the time axis calculated as follows::

            import numpy as np
            t = np.linspace(0, nt * dt, nt + 1)

        where `nt` and `dt` are the number of time step and the time step
        setup in the configuration, respectively.
    """

    def __init__(self, origin, amplitude=1, width=5, evolution=None):
        super().__init__(origin=origin, amplitude=amplitude, width=width)
        self.evolution = None
        self._f = evolution

    def set_evolution(self, t):
        """ Set time evolution of the source.

            Parameters
            ----------
            t : numpy.array
                Time axis
        """
        if isinstance(self._f, (int, float)):
            f = self._f
            self.evolution = self.amplitude * _np.sin(2 * _np.pi * f * t)

        elif callable(self._f):
            self.evolution = self.amplitude * self.evolution(t)