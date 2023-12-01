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
    r"""Pressure source :

    .. math::

        p_{\text{point}} = S_0 e^{- ((x - x_0)^{\beta} + (y - y_0)^{\beta}) / (B_x \delta x))^{\beta}}

    or

    .. math::

        p_{\text{ring}} = S_0 e^{- (\sqrt{(x - x_0)^2 + (y - y_0)^2} + \sqrt{x_r^2 + y_r^2})^{\beta} / (B_x \delta x))^{\beta}}

    References
    ----------

    .. [1] S. Kang et al. « A Physics-Based Approach to Oversample Multi-Satellite, 
        Multi-Species Observations to a Common Grid ». Preprint. Gases/Remote Sensing/Data 
        Processing and Information Retrieval, 23 août 2018. https://doi.org/10.5194/amt-2018-253.

    Parameters
    ----------
    origin: tuple
        Initial position of the pulse :math:`(x_0, y_0[, z_0])`.
    S0: float, optional
        Amplitude :math:`S_0` of the pulse in Pa. 1 by default.
    Bx, By, Bz: int, optional
        Widths :math:`B_x, B_y, B_z` of the pulse in number of spatial steps. 5 by default.
    kx, ky, kz, k: int, optional
        Orders :math:`\beta` of the pulse. Order 2 by default for axis and 1 for global.
    R: float, optional
        Radius following x for annular source
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

    KINDS = ('point', 'ring')

    def __init__(self, origin, S0=1., Bx=5, By=5, Bz=5, kx=2, ky=2, kz=2, k=1, Rx=0, evolution=None):

        if len(origin) not in (2, 3):
            raise ValueError(f'{type(self).__name__}.origin: length 2 or 3 expected')

        self.ndim = len(origin)
        self.origin = origin

        self.S0 = S0
        self.Bx = int(abs(Bx))
        self.By = int(abs(By))
        self.Bz = int(abs(Bz))
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k = k
        self.Rx = Rx
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
        s = f'{self.tag.title()} {type(self).__name__} @ {self.origin} '
        widths = (self.Bx, self.By) if self.ndim == 2 else (self.Bx, self.By, self.Bz)
        orders = (self.kx, self.ky, self.k) if self.ndim == 2 else (self.kx, self.ky, self.kz, self.k)
        s += f'[amplitudes={self.S0}/widths=({widths}/orders={orders}/radius={self.Rx}]'
        return s

    def __repr__(self):
        return self.__str__()


class SourceSet:
    """ Set of pressure sources.

    Sources can be declared as initial conditions or time evolving pressure fluctuations.
    To declare time evolving source, the evolution argument must be provided.

    Sources can be super Gaussian [1] or Gaussian ring sources.

    References
    ----------

    .. [1] S. Kang et al. « A Physics-Based Approach to Oversample Multi-Satellite, 
        Multi-Species Observations to a Common Grid ». Preprint. Gases/Remote Sensing/Data 
        Processing and Information Retrieval, 23 août 2018. https://doi.org/10.5194/amt-2018-253.


    Parameters
    ----------

    origin: tuple or (tuple, )
        Coordinates of the center. Can be a tuple or (tuple, ).
        If a tuple of tuples is provided, all tuples must have the same length.
    S0: float or (float, ). Optional.
        Amplitudes of the sources.
        Parameter amplitude is 1 by default for each source.
        Can be float for a single source or tuple of floats for multiple sources.
    Bx, By, Bz: int or (int, ). Optional.
        Width of the sources in number of points.
        Parameter width is 5 by default for each source.
        Can be positive int for a single source or tuple of positive ints for multiple sources.
    k1, k2, k3, k: int or (int, ). Optional.
        Order of the sources. Must be positive (if not, absolute is taken).
        Parameter order is 2 by default for each source.
        Can be int for a single source or tuple of positive ints for multiple sources.
    Rx: float or (float, ). Optional.
        Radius for the case of annular source. 0 by default.
        Can be float for a single source or tuple of floats for multiple sources.
    on: bool or (bool, ). Optional
        Whether source is on or not.
        Parameter on is False by default for each source.
        Can be bool for a single source or tuple of bool for multiple sources.
    evolution: float/func, or (float/func, ). Optional.
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
    # and a x-width of 5 grid points. The second source is located at (150, 150) with the same 
    # amplitude and a x-width of 10 grid points
    s = SourceSet(origin=((100, 100), (150, 150)), S0=10, Bx=(5, 10))

    """

    def __init__(self, origin, **kwargs):

        self.ndim = kwargs.pop('ndim', 3)
        self._origin = self.parse_origin(origin=origin)
        self._kwargs = {key: value if isinstance(value, tuple) else (value, ) for key, value in kwargs.items()}
        self.update()

    def update(self):
        """Update parameters. """
        self.ics = []
        self.tes = []
        for origin, kwargs in zip(self.origin, self.kwargs):
            if kwargs.pop('on', False) and len(origin):
                self.ics.append(Source(origin, **kwargs))

    def convert_to_2d(self, ax):
        """Convert 3d sources to 2d. ax is the axis where to do the transformation. """
        if self.ndim == 3:
            self.ndim = 2
            for src in self:
                src.convert_to_2d(ax)

    def parse_origin(self, origin):
        """Check if origin is relevant."""
        if not isinstance(origin, tuple):
            raise ValueError('SourceSet.origin: tuple expected')
        try:
            if not all(len(o) in (0, self.ndim) for o in origin):
                raise ValueError(f'{type(self).__name__}.origin: inner tuples of length (0, {self.ndim}) expected')
        except TypeError:
            origin = (origin, )
        return origin

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        value = self.parse_origin(origin=value)
        self._origin = value if isinstance(value[0], tuple) else (value,)
        self.update()

    @property
    def on(self):
        """Report whether sources must be activated or not. """
        return self._kwargs.get('on', ())

    @on.setter
    def on(self, value):
        self._kwargs['on'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def S0(self):
        """Amplitudes of the source. """
        return self._kwargs.get('S0', ())

    @S0.setter
    def S0(self, value):
        self._kwargs['S0'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def Bx(self):
        """x-widths of the source. Will be converted to positive integer if not. """
        return self._kwargs.get('Bx', ())

    @Bx.setter
    def Bx(self, value):
        self._kwargs['Bx'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def By(self):
        """y-widths of the source. Will be converted to positive integer if not. """
        return self._kwargs.get('By', ())

    @By.setter
    def By(self, value):
        self._kwargs['By'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def Bz(self):
        """z-widths of the source. Will be converted to positive integer if not. """
        return self._kwargs.get('Bz', ())

    @Bz.setter
    def Bz(self, value):
        self._kwargs['Bz'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def kx(self):
        """Order of the source following x. Will be converted to positive integer if not. """
        return self._kwargs.get('kx', ())

    @kx.setter
    def kx(self, value):
        self._kwargs['kx'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def ky(self):
        """Order of the source following y. Will be converted to positive integer if not. """
        return self._kwargs.get('ky', ())

    @ky.setter
    def ky(self, value):
        self._kwargs['ky'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def kz(self):
        """Order of the source following z. Will be converted to positive integer if not. """
        return self._kwargs.get('kz', ())

    @kz.setter
    def kz(self, value):
        self._kwargs['kz'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def k(self):
        """Global order of the source. Will be converted to positive integer if not. """
        return self._kwargs.get('k', ())

    @k.setter
    def k(self, value):
        self._kwargs['k'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def Rx(self):
        """Radius in the case of annular source. """
        return self._kwargs.get('Rx', ())

    @Rx.setter
    def Rx(self, value):
        self._kwargs['Rx'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def evolution(self):
        """Time evolution of the sources. """
        return self._kwargs.get('evolution', ())

    @evolution.setter
    def evolution(self, value):
        self._kwargs['evolution'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def kwargs(self):
        """Return a list of dictionnaries providing keyword arguments of the sources. """
        prms = _it.zip_longest(*self._kwargs.values())
        return [{key: value for key, value in zip(self._kwargs.keys(), values) 
                                                if value is not None} for values in prms]

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
        return str(self)


class Flow:
    """ Mean flow source.

    Parameters
    ----------
    kind: str or None, optional
        Kind of the flow. Can be one of the Flow.TYPES.
    components: tuple, optional
        Components of the flow.
    ndim: int, optional
        Can only be 2 or 3 for 2d flow or 3d flow, respectively.
    """
    TYPES = (None, 'mean flow', )

    def __init__(self, kind=None, components=(0, 0, 0), ndim=3):

        self.ndim = ndim
        self._kind = kind if kind in Flow.TYPES else None
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

        if self._kind not in Flow.TYPES:
            raise ValueError(f'{type(self).__name__}.kind: must be in {Flow.TYPES}')

    @property
    def kind(self):
        """Kind of mean flow. """
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value
        self._check()

    @property
    def components(self):
        """Components of the mean flow.

        Note
        ----
        If `components` is modified, the time step `dt` is modified too if `kind` is set to an actual flow.
        """
        return self._components

    @components.setter
    def components(self, value):
        self._components = value
        self._check()

    def __str__(self):
        if self.kind:
            return f"\n[Flow]    {self.kind} {self.components}.\n"
        return f"\n[Flow]    None.\n"


    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":

    ics = SourceSet(origin=())
    print('Test 1: ', ics)
    ics = SourceSet(origin=(1, 2, 3), S0=1, Bx=5, on=(True, ))
    print('Test 2: ', ics)
    ics = SourceSet(origin=((1, 2, 3), (4, 5, 6)), S0=1, Bx=5, on=(True, ))
    print('Test 2: ', ics)