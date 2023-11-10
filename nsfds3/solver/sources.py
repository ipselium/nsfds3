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

    * The `Source` class: Describes an pressure source
    * The `SourceSet` class: Describes a set of pressure sources
    * The `Flow` class: Describes a mean flow
"""

import numpy as _np
import itertools as _it


class Source:
    r""" Pressure source :

    .. math:: 
    
        p_{\text{point}} = S_0 e^{- \alpha ((x - x_0)^{\beta} + (y - y_0)^{\beta}) / (B_x \delta x))^{\beta}}

    or 

    .. math::

        p_{\text{ring}} = S_0 e^{- \alpha (\sqrt{(x - x_0)^2 + (y - y_0)^2} + \sqrt{x_r^2 + y_r^2})^{\beta} / (B_x \delta x))^{\beta}}

    Parameters
    ----------
    origin: tuple
        Initial position of the pulse :math:`(x_0, y_0[, z_0])`.
    radius: tuple, optional
        Radius of the source for ring spacial distribution.
    amplitude: float, optional
        Amplitude :math:`S_0` of the pulse in Pa. 1 by default.
    width: int, optional
        Width :math:`B_x` of the pulse in number of spatial steps. 5 by default.
    order: int, optional
        Order :math:`\beta` of the pulse. Order 2 by default.
    alpha: float, optional
        Spacial decrease :math:`\alpha` of the pulse. ln(2) by default.
    stype: str, optional
        Can be either "point" or ring". "point" by default.
    evolution : float or func, optional
        Time evolution of the source.
        If evolution is None, the source will be an initial condition.
        If evolution is a float, it will describe the frequency of a sinusoidal time evolution.
        If evolution is a function, the time evolution of the source will be the result of
        `evolution(t)` where `t` is the time axis calculated as follows::

            import numpy as np
            t = np.linspace(0, nt * dt, nt + 1)

        where `nt` and `dt` are the number of time step and the time step
        setup in the configuration, respectively.
    """

    TYPES = ('point', 'ring')

    def __init__(self, origin, radius=None, amplitude=1., width=5, order=2, alpha=_np.log(2), stype='point', evolution=None):

        if len(origin) not in (2, 3):
            raise ValueError(f'{type(self).__name__}.origin: length 2 or 3 expected')

        self.ndim = len(origin)
        self.origin = origin
        if not radius:
            self.radius = self.ndim * (0., )
        elif len(origin) != len(radius):
            raise ValueError(f'{type(self).__name__}.radius: length {self.ndim} expected')
        else:
            self.radius = radius

        self.amplitude = amplitude
        self.width = int(abs(width))
        self.order = int(abs(order))
        self.alpha = abs(alpha)
        self.stype = stype if stype in self.TYPES else 'point'
        self.evolution = None
        self._f = evolution

    @property
    def tag(self):
        """ Report whether the source is initial or temporal. """
        return 'temporal' if self._f is not None else 'initial'

    def convert_to_2d(self, ax):
        """ Convert 3d source to 2d. """
        if self.ndim == 3:
            self.ndim = 2
            self.origin = tuple(s for i, s in enumerate(self.origin) if i != ax)
            self.radius = tuple(s for i, s in enumerate(self.radius) if i != ax)

    def set_evolution(self, t):
        """ Set time evolution of the source. 
        If this parameter is not set, the source is an initial condition

        Parameter
        ---------
        t : numpy.array
            Time axis
        """
        if isinstance(self._f, (int, float)):
            self.evolution = self.amplitude * _np.sin(2 * _np.pi * self._f * t)
        elif callable(self._f):
            self.evolution = self.amplitude * self.evolution(t)

    def __str__(self):
        s = f'{self.tag.title()} {self.stype} {type(self).__name__} @ {self.origin} '
        if self.stype == 'ring':
            s += f'with radius {self.radius} '
        s += f'[amplitudes={self.amplitude}/widths={self.width}/order={self.order}/alpha={self.alpha:.2f}]'
        return s

    def __repr__(self):
        return self.__str__()


class SourceSet:
    """ Set of pressure sources. 

    Sources can be declared as initial conditions or time evolving pressure fluctuations.
    To declare time evolving source, the evolution argument must be provided.

    Parameters
    ----------
    
    origins: tuple or (tuple, )
        Coordinates of the center. Can be a tuple or (tuple, ).
        If a tuple of tuples is provided, all tuples must have the same length.
    radii: tuple or (tuple, ). Optional.
        Radii of the sources used for ring source type. Can be a tuple or (tuple, ).
        Parameter radius is (0., 0.[, 0.]) by default for each source.
        If a tuple of tuples is provided, all tuples must have the same length. 
        Note that for now, only circular rings can be generated, not ellipsoidal ones.
    amplitudes: float or (float, ). Optional.
        Amplitudes of the sources.
        Parameter amplitude is 1 by default for each source.
        Can be float for a single source or tuple of floats for multiple sources.
    widths: int or (int, ). Optional.
        Width of the sources. 
        Parameter width is 5 by default for each source.
        Can be int for a single source or tuple of positive ints for multiple sources.
    orders: int or (int, ). Optional.
        Order of the sources. Must be positive (if not, absolute is taken). 
        Parameter order is 2 by default for each source.
        Can be int for a single source or tuple of positive ints for multiple sources.
    alphas: float or (float, ). Optional.
        Decrease of the sources. Must be positive (if not, absolute is taken). 
        Parameter alpha is log(2) by default for each source.
        Can be float for a single source or tuple of floats for multiple sources.
    types: str or (str, ). Optional.
        Type of the sources. Type can be "ring" or "point".
        Parameter type is 'point' by default for each source.
        Can be str for a single source or tuple of strings for multiple sources.
    on: bool or (bool, ). Optional
        Whether source is on or not. 
        Parameter on is False by default for each source.
        Can be bool for a single source or tuple of bool for multiple sources.
    evolutions: float/func, or (float/func, ). Optional.
        Time evolution of the source.
        Parameter evolution is None by default for each source.
        If evolution is None, the source will be an initial condition.
        If evolution is a float, it will describe the frequency of a sinusoidal time evolution.
        If evolution is a function, the time evolution of the source will be the result of
        `evolution(t)` where `t` is the time axis calculated as follows::

            import numpy as np
            t = np.linspace(0, nt * dt, nt + 1)

        where `nt` and `dt` are the number of time step and the time step
        setup in the configuration, respectively.

    Example
    -------
    # Declare 2 initial conditions, the first one located at (100, 100) with an amplitude of 10
    # and a width of 5. The second source is located at (150, 150) with the same amplitude
    # and a width of 10
    s = SourceSet(origins=((100, 100), (150, 150)), amplitudes=10, widths=(5, 10))

    """

    KEYS = ('origin', 'radius', 'amplitude', 'width', 'order', 'alpha', 'stype', 'on', 'evolution')

    def __init__(self, origins, radii=(), amplitudes=(), widths=(), orders=(), alphas=(), types=(), on=(), evolutions=(), ndim=None):
        
        self.ndim = ndim
        self._on = on
        self._origins = origins
        self._radii = radii
        self._widths = widths
        self._amplitudes = amplitudes
        self._orders = orders
        self._alphas = alphas
        self._types = types
        self._evolutions = evolutions
        self._update()

    def convert_to_2d(self, ax):
        """ Convert 3d sources to 2d. ax is the axis where to do the transformation. """
        if self.ndim == 3:
            self.ndim = 2
            for src in self:
                src.convert_to_2d(ax)

    def _update(self):
        """ Update parameters. """
        self.ics = []
        self.tes = []
        self._check()
        for kwargs in self.kwargs:
            if kwargs.pop('on', False):
                if kwargs.get('evolution', None) is None:
                    self.ics.append(Source(**kwargs))
                else:
                    self.tes.append(Source(**kwargs))

    def _check(self):
        """ Check that parameters are consistents. """

        # Check origins
        if not isinstance(self._origins, (tuple, list)):
            raise ValueError(f'{type(self).__name__}.origins: tuple expected')
        
        if not any(isinstance(o, (tuple, list)) for o in self._origins):
            self._origins = self._origins,

        if self.ndim is None:
            self.ndim = len(self._origins[0])

        if not all(len(o) in (0, self.ndim) for o in self._origins):
            raise ValueError(f'{type(self).__name__}.origins: inner tuples of length {self.ndim} expected')

        # Check radii
        if not isinstance(self._radii, (tuple, list)):
            raise ValueError(f'{type(self).__name__}.radii: tuple expected')

        if not any(isinstance(o, (tuple, list)) for o in self._radii):
            self._radii = self._radii,
        else:
            self._radii = self._radii[:min(len(self._radii), len(self._origins))]

        if not all(len(o) in (0, self.ndim) for o in self._radii):
            raise ValueError(f'{type(self).__name__}.radii: inner tuples of length {self.ndim} expected')

        # Other parameters
        prms = ('_amplitudes', '_widths', '_orders', '_alphas', '_types', '_on', '_evolutions')
        for prm in prms:
            value = getattr(self, prm)
            if not isinstance(value, (tuple, list)):
                setattr(self, prm, (value, ))
            else:
                setattr(self, prm, value[:min(len(value), len(self._origins))])

        self._on = tuple(bool(v) for v in self._on)

    @property
    def kwargs(self):
        """ Return a list of dictionnaries providing keyword arguments of the sources. """
        prms = _it.zip_longest(self.origins, self.radii, self.amplitudes, self.widths, 
                               self.orders, self.alphas, self.types, self.on, self.evolutions)
        return [{key: value for key, value in zip(self.KEYS, values) if value is not None} for values in prms]

    @property
    def origins(self):
        """ Origins of the source. """
        return self._origins

    @origins.setter
    def origins(self, value):
        self._origins = value
        self._update()

    @property
    def on(self):
        """ Report whether sources must be activated or not. """
        return self._on
    
    @on.setter
    def on(self, value):
        self._on = value
        self._update()

    @property
    def radii(self):
        """ Radii of the ring sources. """
        return self._radii

    @radii.setter
    def radii(self, value):
        self._radii = value
        self._update()

    @property
    def amplitudes(self):
        """ Amplitudes of the source. """
        return self._amplitudes

    @amplitudes.setter
    def amplitudes(self, value):
        self._amplitudes = value
        self._update()

    @property
    def widths(self):
        """ Widths of the source. Will be converted to positive integer if not. """
        return self._widths

    @widths.setter
    def widths(self, value):
        self._widths = value
        self._update()

    @property
    def orders(self):
        """ Order of the source. Will be converted to positive integer if not. """
        return self._orders

    @orders.setter
    def orders(self, value):
        self._orders = value
        self._update()

    @property
    def alphas(self):
        """ Decreasing of the source. Will be converted in positive float if not.  """
        return self._alphas

    @alphas.setter
    def alphas(self, value):
        self._alphas = value
        self._update()

    @property
    def types(self):
        """ Type of the source. Can be "ring" or "point". """
        return self._types

    @types.setter
    def types(self, value):
        self._types = value
        self._update()

    @property
    def evolutions(self):
        """ Time evolutions of the sources. """
        return self._evolutions

    @evolutions.setter
    def evolutions(self, value):
        self._evolutions = value
        self._update()

    def __iter__(self):
        return iter(self.tes + self.ics)

    def __len__(self):
        return len(self.tes) + len(self.ics)

    def __str__(self):
        if not self:
            s = '\n[Sources] None'
        else:
            s = '\n[Sources]'
            for src in self:
                s += f"\n\t- {src}."
        return s

    def __repr__(self):
        return self.__str__()


class Flow:
    """ Mean flow source. 
    
    Parameters
    ----------
    ftype: str or None, optional
        Type of the flow. Can be one of the Flow.TYPES.
    components: tuple, optional
        Components of the flow.
    ndim: int, optional
        Can only be 2 or 3 for 2d flow or 3d flow, respectively.
    """
    TYPES = (None, 'mean flow', )

    def __init__(self, ftype=None, components=(0, 0, 0), ndim=3):

        self.ndim = ndim
        self._ftype = ftype if ftype in Flow.TYPES else None
        self._components = components
        self._check()

    def convert_to_2d(self, ax):
        """ Convert 3d mean flow to 2d. ax is the axis where to do the transformation. """
        if self.ndim == 3:
            self.ndim = 2
            self.components = tuple(s for i, s in enumerate(self.components) if i != ax)

    def _check(self):
        """ Check that flow parameters are set correctly. """
        if not isinstance(self._components, tuple):
            raise ValueError(f'{type(self).__name__}.components: tuple expected')

        if len(self._components) != self.ndim:
            raise ValueError(f'{type(self).__name__}.components: {self.ndim}-length tuple expected')

        if self._ftype not in Flow.TYPES:
            raise ValueError(f'{type(self).__name__}.ftype: must be in {Flow.TYPES}')

    @property
    def ftype(self):
        """ Type of mean flow. """
        return self._ftype

    @ftype.setter
    def ftype(self, value):
        self._ftype = value
        self._check()

    @property
    def components(self):
        """ Components of the mean flow.

        Note
        ----
        If `components` is modified, the time step `dt` is modified too if `ftype` is set to an actual flow.
        """
        return self._components

    @components.setter
    def components(self, value):
        self._components = value
        self._check()

    def __str__(self):
        if self.ftype:
            return f"\n[Flow]    {self.ftype} {self.components}.\n"
        return f"\n[Flow]    None.\n"


    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":

    ics = SourceSet(origins=())
    print('Test 1: ', ics)
    ics = SourceSet(origins=(1, 2, 3), amplitudes=1, widths=5, orders=2, alphas=1, on=(True, ), types='ring')
    print('Test 2: ', ics)
    ics = SourceSet(origins=((1, 2, 3), (4, 5, 6)), amplitudes=1, widths=5, orders=2, alphas=1, on=(True, ), types=('point', 'ring'))
    print('Test 2: ', ics)    