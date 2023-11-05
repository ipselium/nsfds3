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
import itertools as _it


class Pulse:
    """ Pressure Gaussian pulse as initial condition :

    P = amplitude * exp(- alpha * ((x - x0)**order + (y - y0)**order) / (width * dx)**order )

    Parameters
    ----------
    origin: tuple
        Initial position of the pulse.
    radius: tuple
        Radius of the source for ring spacial distribution.
    amplitude: float, optional
        Amplitude of the pulse in Pa. 1 by default.
    width: int, optional
        Width of the pulse in number of spatial steps. 5 by default.
    order: int, optional
        Order of the pulse. Order 2 by default.
    alpha: float, optional
        Spacial decrease of the pulse. ln(2) by default.
    stype: str, optional
        Can be either "point" or ring". "point" by default.
    """

    def __init__(self, origin, radius=None, amplitude=1, width=5, order=2, alpha=_np.log(2), stype='point'):

        self.origin = origin
        self.radius = radius
        self.amplitude = 1. if amplitude is None else float(amplitude)
        self.width = 5 if width is None else int(abs(width))
        self.order = 2. if order is None else float(abs(order))
        self.alpha = _np.log(2) if alpha is None else float(abs(alpha))
        self.stype = stype if stype in ('ring', 'point') else 'point'

    def __str__(self):
        return f'{self.stype.title()} {type(self).__name__} @ {self.origin} [S0={self.amplitude}/B0={self.width}/order={self.order}]'

    def __repr__(self):
        return self.__str__()


class Monopole(Pulse):
    """ Gaussian source evolving in time.

    Parameters
    ----------
    origin : tuple
        Position of the source.
    radius: tuple
        Radius of the source for ring spacial distribution.
    amplitude: float, optional
        Amplitude of the pulse in Pa. 1 by default.
    width: int, optional
        Width of the pulse in number of spatial steps. 5 by default.
    order: int, optional
        Order of the pulse. Order 2 by default.
    alpha: float, optional
        Spacial decrease of the pulse. ln(2) by default.
    stype: str, optional
        Can be either "point" or ring". "point" by default.
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

    def __init__(self, origin, radius=None, amplitude=1, width=5, order=2, alpha=_np.log(2), stype="point", evolution=None):
        super().__init__(origin=origin, radius=radius, amplitude=amplitude, width=width, order=order, alpha=alpha, stype=stype)
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


class ICS:
    """ Setup Initial Conditions. 
    
    origins: tuple or (tuple, )
        Coordinates of the center. Can be a tuple or (tuple, ).
        If a tuple of tuples is provided, all must have the same length.
    S0: float or (float, )
        Amplitudes of the sources. 
        Can be float for a single source or tuple of floats for multiple sources.
    B0: int or (int, )
        Width of the sources.
        Can be int for a single source or tuple of ints for multiple sources.
    orders: int or (int, )
        Order of the sources. Must be positive (if not, absolute is taken). 
        Can be int for a single source or tuple of ints for multiple sources.
    alphas: float or (float, )
        Decrease of the sources. Must be positive (if not, absolute is taken). 
        Can be int for a single source or tuple of ints for multiple sources.
    stypes: str of (str, )    
        Type of the sources. Type can be "ring" or "point".
        Can be str for a single source or tuple of ints for multiple sources.        
    """
    def __init__(self, origins, S0, B0, orders, alphas, stypes):
        
        self._on = True
        self._origins = origins
        self._B0 = B0
        self._S0 = S0
        self._orders = orders
        self._alphas = alphas
        self._stypes = stypes
        self._update()

    def _update(self):
        """ Update parameters. """
        self.elements = []
        self._check()
        if self.on and all(self.origins):
            for o, s, b, n, a, t in _it.zip_longest(self.origins, self.S0, self.B0, self.orders, self.alphas, self.stypes):
                self.elements.append(Pulse(o, s, b, n, a, t))

    def _check(self):
        """ Check that parameters are consistents. """
        if not any(isinstance(o, (tuple, list)) for o in self._origins):
            self._origins = self._origins,

        if not isinstance(self._S0, (tuple, list)):
            self._S0 = self._S0,

        if not isinstance(self._B0, (tuple, list)):
            self._B0 = self._B0,
        
        if not isinstance(self._orders, (tuple, list)):
            self._orders = self._orders,
        
        if not isinstance(self._alphas, (tuple, list)):
            self._alphas = self._alphas,
        
        if not isinstance(self._stypes, (tuple, list)):
            self._stypes = self._stypes,
        
        if not all(self.origins):
            self._on = False
        elif not all([len(o) in (2, 3) for o in self._origins]):
            raise ValueError(f'[{type(self).__name__}].origin: each element must be of len 2 or 3')

    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        self._on = bool(value)
        self._update()

    @property
    def origins(self):
        """ Origins of the source. """
        return self._origins

    @origins.setter
    def origins(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].origin: tuple expected')
        self._origins = value
        self._update()

    @property
    def S0(self):
        """ Amplitudes of the source. """
        return self._S0

    @S0.setter
    def S0(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].S0: tuple expected')
        self._S0 = value
        self._update()

    @property
    def B0(self):
        """ Widths of the source. """
        return self._B0

    @B0.setter
    def B0(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].B0: tuple expected')
        self._B0 = value
        self._update()

    @property
    def orders(self):
        """ Order of the source. """
        return self._orders

    @orders.setter
    def orders(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].orders: tuple expected')
        self._orders = value
        self._update()

    @property
    def alphas(self):
        """ Decreasing of the source. """
        return self._alphas

    @alphas.setter
    def alphas(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].alphas: tuple expected')
        self._alphas = value
        self._update()

    @property
    def stypes(self):
        """ Type of the source. Can be "ring" or "ponctual". """
        return self._stypes

    @stypes.setter
    def stypes(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].stypes: tuple expected')
        self._stypes = value
        self._update()

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        s = ''
        for src in self:
            s += f"\t- {src}.\n"
        return s

    def __repr__(self):
        return self.__str__()
    

class SRC(ICS):

    def __init__(self, origins, S0, B0, orders, alphas, stypes, evolutions):
        
        self._evolutions = evolutions
        super().__init__(origins, S0, B0, orders, alphas, stypes)

    def _check(self):
        """ Check that parameters are consistents. """

        super()._check()
        
        if isinstance(self._evolutions, (int, float, str)):
            self._evolutions = self._evolutions,
        
    def _update(self):
        """ Update parameters. """
        self.elements = []
        if self.on and all(self.origins):
            self._check()
            for o, s, b, n, a, t, e in _it.zip_longest(self.origins, self.S0, self.B0, self.orders, self.alphas, self.stypes, self.evolutions):
                self.elements.append(Monopole(o, s, b, n, a, t, e))

    @property
    def evolutions(self):
        """ Time evolutions of the sources. """
        return self._evolutions

    @evolutions.setter
    def evolutions(self, value):
        if not isinstance(value, tuple):
            raise ValueError(f'[{type(self).__name__}].evolution: tuple expected')
        self._evolutions = value
        self._update()