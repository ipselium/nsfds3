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

    * The `CustomInitialConditions` class: Describes initial conditions
    * The `Source` class: Describes an pressure source
    * The `SourceSet` class: Describes a set of pressure sources
    * The `Flow` class: Describes a mean flow
"""

import sys
import numpy as _np
import matplotlib.pyplot as _plt
import itertools as _it
from rich import print
from rich.prompt import IntPrompt, Prompt
from nsfds3.graphics.utils import cmap_jet, MidPointNorm
from nsfds3.utils.data import DataExtractor
from nsfds3.utils.files import get_objects
from nsfds3.utils.misc import Confirm
from libfds.cmaths import super_gaussian2d, super_gaussian3d


class CustomInitialConditions:
    """Customize initial conditions for acoustic pressure and/or velocities.

    Parameters
    ----------
    cfg: CfgSetup
        Configuration of the simulation
    msh: CartesianGrid, CurvilinearGrid
        Grid used for the simulation

    Notes
    -----
    To setup custom initial conditions, one must override the
    `CustomInitialConditions.__post_init__` method.
    The following attributes can be customized:

            * `self.p`: acoustic pressure (without static pressure)
            * `self.vx`: x-component of velocity
            * `self.vy`: y-component of velocity
            * `self.vz`: z-component of velocity
    Example
    -------

    ::

        class Ring(CustomInitialConditions):

            def __post_init__(self):
                origin = 100, 100
                k = 12
                self.p = self.super_gaussian(origin, k=12)
    """

    varnames = ('p', 'vx', 'vy', 'vz')

    def __init__(self, cfg, msh):

        self.cfg = cfg
        self.msh = msh

        self.p = None
        self.vx = None
        self.vy = None
        self.vz = None

        self._has_old_conf = self.check_older_run()
        if self._has_old_conf:
            self._prompt_for_resume()

        if not self.has_old_fields:
            self.__post_init__()
            self.check()

    def __post_init__(self):
        """One can override this method to provide custom initial condition for:

            * acoustic pressure (self.p) (without static pressure !)
            * x-component of velocity (self.vx)
            * y-component of velocity (self.vy)
            * z-component of velocity (self.vz)

            Note that super_gaussian() method is available here for convenience.
        """
        pass

    def check_older_run(self):
        """Report whether there is an older run or not."""
        if self.cfg.files.data_path.is_file():
            cfg, _ = get_objects(self.cfg.files.directory, self.cfg.files.name)
            if cfg == self.cfg and cfg.sol.itmax !=0:
                if not self.cfg.quiet and (self.cfg.sol.resume or cfg.sol.itmax < cfg.sol.nt):
                    print(f'[bold bright_magenta]Previous run found for {cfg.cfgfile.stem} until iteration {cfg.sol.itmax}/{cfg.sol.nt}.')
                return True
        return False

    def _prompt_for_resume(self):
        cfg, _ = get_objects(self.cfg.files.directory, self.cfg.files.name)
        if cfg.sol.itmax < self.cfg.sol.nt:
            if not self.cfg.sol.resume:
                resume = Confirm.ask(f'[bold bright_magenta][blink]Continue last simulation[/] (if not simulation will start from scratch) ? ')
            else:
                resume = True
        elif self.cfg.sol.resume:
            self.cfg.sol.nt = int(max(cfg.sol.itmax, self.cfg.sol.nt))
            nt = IntPrompt.ask(f'[bold bright_magenta][blink]Number of additional iterations? ', default=0)
            self.cfg.sol.nt += nt
            resume = True
        else:
            resume = False

        if resume:
            self.resume()

    def resume(self):
        """Setup fields to resume last simulation with the current config.
        Set r, ru, rv[, rw], re instance attributes to the values at the last 
        iteration itmax reached during last simulation."""
        if not self._has_old_conf:
            print('[bold bright_magenta]Old configuration not found.')
            return

        with DataExtractor(self.cfg.files.data_path) as data:
            self.cfg.sol.itmax = data.get_attr('itmax')
            self._r = data.get_dataset(f'r_it{self.cfg.sol.itmax}')
            self._ru = data.get_dataset(f'ru_it{self.cfg.sol.itmax}')
            self._rv = data.get_dataset(f'rv_it{self.cfg.sol.itmax}')
            self._re = data.get_dataset(f're_it{self.cfg.sol.itmax}')
            if self.msh.ndim == 3:
                self._rw = data.get_dataset(f'rw_it{self.cfg.sol.itmax}')
        self.cfg.overwrite()
        if not self.cfg.quiet:
            print(f'[bold bright_cyan]Fields at iteration {self.cfg.sol.itmax}/{self.cfg.sol.nt} loaded!')

    def super_gaussian(self, origin, kx=2, ky=2, kz=2, k=1,
                                     Bx=None, By=None, Bz=None, Rx=0):

        if len(origin) != self.msh.ndim:
            raise ValueError(f'origin must be length {self.msh.origin} tuple')

        Bx = Bx if Bx else self.msh.dx * 5
        By = By if By else Bx

        if self.msh.ndim == 3:
            Bz = Bz if Bz else Bx
            return super_gaussian3d(*self.msh.paxis,
                                    *origin, kx, ky, kz, k, Bx, By, Bz, Rx)

        return super_gaussian2d(*self.msh.paxis, *origin, kx, ky, k, Bx, By, Rx)

    def check(self):
        """Check that initial conditions are valid ones."""
        for varname, var in self.vars.items():
            if not isinstance(var, _np.ndarray) or var.shape != self.msh.shape:
                raise ValueError(f'{varname} must be a numpy.ndarray of dim {self.msh.shape}')

    @property
    def custom(self):
        """Report whether initial conditions are provided or not."""
        return bool(self.vars)

    @property
    def has_old_fields(self):
        """Report whether old fields have been initialized or not."""
        return hasattr(self, '_r') and hasattr(self, '_re')

    @property
    def vars(self):
        """Return initial conditions that have been provided."""
        return {varname:getattr(self, varname) for varname in self.varnames
                if getattr(self, varname) is not None}

    def show(self):
        """Show initial conditions graphically."""
        number = len(self.vars)
        if not number:
            print('[bold bright_magenta]Nothing to show')
        else:
            self._show(number)

    def _show(self, number):
        cmap = cmap_jet()
        fig, axes = _plt.subplots(1, number, tight_layout=True)
        if number == 1:
            axes = _np.array([axes, ])
        for ax, (varname, value) in zip(axes, self.vars.items()):
            norm = MidPointNorm(vmin=value.min(), vmax=value.max(), midpoint=0)
            ax.pcolormesh(self.msh.x, self.msh.y, value.T, cmap=cmap, norm=norm)
            ax.set_title(varname)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_aspect('equal')
        _plt.show()

    def __str__(self):
        if self.custom:
            vars = [varname for varname in self.varnames if isinstance(getattr(self, varname), _np.ndarray)]
            vars = '/'.join(vars)
            return f'ics : {self.custom} ({vars})'
        return f'ics : {self.custom}'

    def __repr__(self):
        return str(self)


CustomInitialConditions.super_gaussian.__doc__ = super_gaussian3d.__doc__


class Source:
    r"""Parameters for Super Gaussian Acoustic sources. This class is not 
    intended to be instantiated or modified directly. It is used by SourceSet to
    set a set of source parameters.

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
        Widths :math:`B_x, B_y, B_z` of the pulse. 5 by default.
    kx, ky, kz, k: int, optional
        Orders :math:`\beta` of the pulse. Order 2 by default for axis and 1 for global.
    Rx: float, optional
        Radius following x for annular source
    evolution: float/func, or (float/func, ). Optional.
        Time evolution of the source.
    """

    def __init__(self, origin, S0=1., Bx=0.1, By=0.1, Bz=0.1, kx=2, ky=2, kz=2, k=1,
                 Rx=0, evolution=None):

        self._origin = origin
        self._S0 = S0
        self._Bx = Bx
        self._By = By
        self._Bz = Bz
        self._kx = kx
        self._ky = ky
        self._kz = kz
        self._k = k
        self._Rx = Rx
        self._evolution = None
        self._f = evolution

    @property
    def tag(self):
        """Report whether the source is initial or temporal."""
        return 'temporal' if self._f is not None else 'initial'

    @property
    def origin(self):
        """Read-only origin of the source."""
        return self._origin

    @property
    def S0(self):
        """Read-only amplitude of the source."""
        return self._S0

    @property
    def Bx(self):
        """Read-only x-width of the source."""
        return self._Bx

    @property
    def By(self):
        """Read-only y-width of the source."""
        return self._By

    @property
    def Bz(self):
        """Read-only z-width of the source."""
        return self._By

    @property
    def kx(self):
        """Read-only x-order of the source."""
        return self._kx

    @property
    def ky(self):
        """Read-only y-order of the source."""
        return self._ky

    @property
    def kz(self):
        """Read-only z-order of the source."""
        return self._kz

    @property
    def k(self):
        """Read-only global order of the source."""
        return self._k

    @property
    def Rx(self):
        """Read-only radius of the source."""
        return self._Rx

    def set_evolution(self, t):
        """Set time evolution of the source.
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
        s = f'{self.tag.title()} Source @ {self.origin} '
        widths = (self.Bx, self.By) if len(self.origin) == 2 else (self.Bx, self.By, self.Bz)
        orders = (self.kx, self.ky, self.k) if len(self.origin) == 2 else (self.kx, self.ky, self.kz, self.k)
        s += f'[amplitudes={self.S0}/widths={widths}/orders={orders}/radius={self.Rx}]'
        return s

    def __repr__(self):
        return self.__str__()


class SourceSet:
    r"""Set of pressure sources in Super Gaussian form.

    Super Gaussian [1] can be declared as initial conditions or time evolving pressure fluctuations.
    To declare time evolving source, the evolution argument must be provided.


    .. math::

        p_{\text{ring}} = S_0 e^{- (\sqrt{(x - x_0)^2 + (y - y_0)^2} + \sqrt{x_r^2 + y_r^2})^{\beta} / (B_x \delta x))^{\beta}}


    References
    ----------

    .. [1] S. Kang et al. « A Physics-Based Approach to Oversample Multi-Satellite,
        Multi-Species Observations to a Common Grid ». Preprint. Gases/Remote Sensing/Data
        Processing and Information Retrieval, 23 août 2018. https://doi.org/10.5194/amt-2018-253.


    Parameters
    ----------
    shape: tuple
        Shape of the computational domain.
    origin: tuple or (tuple, )
        Coordinates of the center. Can be a tuple or (tuple, ).
        If a tuple of tuples is provided, all tuples must have the same length.
    flat: tuple or None
        Whether geometry is flattened or not.
        If a tuple is provided, it must provide (flat axis, flat index).
    S0: float or (float, ). Optional.
        Amplitude :math:`S_0` of the pulse in Pa. 1 by default.
        Parameter amplitude is 1 by default for each source.
        Can be float for a single source or tuple of floats for multiple sources.
    Bx, By, Bz: float or (float, ). Optional.
        Width :math:`B_x, B_y, B_z` of the sources.
        Parameter width is 0.1 by default for each source.
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
    # and a x-width of 0.1 grid points. The second source is located at (150, 150) with the same
    # amplitude and a x-width of 0.2 grid points
    s = SourceSet(origin=((100, 100), (150, 150)), S0=10, Bx=(0.1, 0.2))

    """

    def __init__(self, shape, origin, flat=None, **kwargs):

        self._shape = shape
        self._ndim = len(self._shape)
        self._flat = flat
        if self._ndim != 3 and self._flat:
            raise ValueError("SourceSet: length 3 expected for shape")

        self._origin = self._parse_origin(origin=origin)
        self._kwargs = {key: value if isinstance(value, tuple) else (value, ) for key, value in kwargs.items()}
        self.update()

    def update(self):
        """Update parameters."""
        self.ics = []
        self.tes = []
        for origin, kwargs in zip(self.origin, self.kwargs):
            if kwargs.pop('on', False) and len(origin):
                self.ics.append(Source(origin, **kwargs))

    def _parse_origin(self, origin):
        """Parse origin parameter."""
        # is tuple ?
        if not isinstance(origin, tuple):
            raise ValueError('SourceSet.origin: tuple expected')

        # is tuple of tuples ?
        if not any(isinstance(p, (tuple, list)) for p in origin):
            origin = origin,
        origin = tuple(tuple(c) for c in origin)

        # is length ok and location in the domain ?
        if hasattr(self, '_origin'):
            ndims = (2, 3) if self._flat else (self._ndim, )
        else:
            ndims = (self._ndim, )
        for i, loc in enumerate(origin):
            if len(loc) not in ndims:
                raise ValueError(f"SourceSet.origin: tuples of length {'|'.join(str(c) for c in ndims)} expected")
            if any(not 0 <= c < s for c, s in zip(loc, self._shape)):
                raise ValueError(f'SourceSet.origin[{i}]: out of bounds')
        return origin

    @property
    def origin(self):
        if self._flat:
            ax, _ = self._flat
            return tuple(tuple(c for i, c in enumerate(loc) if i != ax) for loc in self._origin)
        return self._origin

    @origin.setter
    def origin(self, value):
        value = self._parse_origin(origin=value)
        if self._flat and any(len(c) == 2 for c in value):
            ax, idx = self._flat
            value = tuple(c[:ax] + (idx, ) + c[ax:] for c in value)
        self._origin = value
        self.update()

    @property
    def on(self):
        """Report whether sources must be activated or not."""
        return self._kwargs.get('on', ())

    @on.setter
    def on(self, value):
        self._kwargs['on'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def S0(self):
        """Amplitudes of the source."""
        return self._kwargs.get('S0', ())

    @S0.setter
    def S0(self, value):
        self._kwargs['S0'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def Bx(self):
        """x-widths of the source. Will be converted to positive integer if not."""
        return self._kwargs.get('Bx', ())

    @Bx.setter
    def Bx(self, value):
        self._kwargs['Bx'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def By(self):
        """y-widths of the source. Will be converted to positive integer if not."""
        return self._kwargs.get('By', ())

    @By.setter
    def By(self, value):
        self._kwargs['By'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def Bz(self):
        """z-widths of the source. Will be converted to positive integer if not."""
        return self._kwargs.get('Bz', ())

    @Bz.setter
    def Bz(self, value):
        self._kwargs['Bz'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def kx(self):
        """Order of the source following x. Will be converted to positive integer if not."""
        return self._kwargs.get('kx', ())

    @kx.setter
    def kx(self, value):
        self._kwargs['kx'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def ky(self):
        """Order of the source following y. Will be converted to positive integer if not."""
        return self._kwargs.get('ky', ())

    @ky.setter
    def ky(self, value):
        self._kwargs['ky'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def kz(self):
        """Order of the source following z. Will be converted to positive integer if not."""
        return self._kwargs.get('kz', ())

    @kz.setter
    def kz(self, value):
        self._kwargs['kz'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def k(self):
        """Global order of the source. Will be converted to positive integer if not."""
        return self._kwargs.get('k', ())

    @k.setter
    def k(self, value):
        self._kwargs['k'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def Rx(self):
        """Radius in the case of annular source."""
        return self._kwargs.get('Rx', ())

    @Rx.setter
    def Rx(self, value):
        self._kwargs['Rx'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def evolution(self):
        """Time evolution of the sources."""
        return self._kwargs.get('evolution', ())

    @evolution.setter
    def evolution(self, value):
        self._kwargs['evolution'] = tuple(value) if hasattr(value, '__iter__') else (value,)
        self.update()

    @property
    def kwargs(self):
        """Return a list of dictionnaries providing keyword arguments of the sources."""
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
    """Mean flow source.

    Parameters
    ----------
    shape: tuple
        Shape of the computational domain.
    ftype: str or None, optional
        Type of the flow. Can be one of the Flow.TYPES.
    components: tuple, optional
        Components of the flow.
    flat: tuple or None, optional
        Whether geometry is flattened or not. 
        If a tuple is provided, it must provide (flat axis, flat index).
    """
    TYPES = (None, 'mean flow', )

    def __init__(self, shape, ftype=None, components=(0, 0, 0), ndim=None, flat=None):

        self._shape = shape
        self._ndim = len(shape)
        self._flat = flat
        if self._ndim != 3 and self._flat:
            raise ValueError("Flow: with flat, ndim must be 3")

        self._ftype = self._parse_ftype(ftype)
        self._components = self._parse_components(components)

    @property
    def ftype(self):
        """Type of the mean flow."""
        return self._ftype

    @ftype.setter
    def ftype(self, value):
        value = self._parse_ftype(value)
        self._ftype = value

    @property
    def components(self):
        """Components of the mean flow.

        Note
        ----
        If `components` is modified, the time step `dt` is modified too if `ftype` is set to an actual flow.
        """
        if self._flat:
            ax, _ = self._flat
            return tuple(s for i, s in enumerate(self._components) if i != ax)
        return self._components

    @components.setter
    def components(self, value):
        value = self._parse_components(value)
        if self._flat and len(value) == 2:
            ax, _ = self._flat
            value = value[:ax] + (self._components[ax], ) + value[ax:]
        self._components = value

    def _parse_components(self, value):
        """Check that flow parameters are set correctly."""
        if not isinstance(value, tuple):
            raise ValueError(f'Flow.components: tuple expected')

        if hasattr(self, '_components'):
            ndims = (2, 3) if self._flat else (self._ndim, )
        else:
            ndims  = (self._ndim, )
        if len(value) not in ndims:
            raise ValueError(f"Flow.components: {'|'.join(str(c) for c in ndims)}-length tuple expected")

        return value

    def _parse_ftype(self, value):
        """Parse flow type."""
        if isinstance(value, str):
            value = value.lower()
        if value not in Flow.TYPES:
            raise ValueError(f'{type(self).__name__}.ftype: must be in {Flow.TYPES}')
        return value

    def __str__(self):
        if self.ftype:
            return f"\n[Flow]    {self.ftype} {self.components}.\n"
        return f"\n[Flow]    None.\n"

    def __repr__(self):
        return str(self)


if __name__ == "__main__":

    ics = SourceSet(origin=())
    print('Test 1: ', ics)
    ics = SourceSet(origin=(1, 2, 3), S0=1, Bx=5, on=(True, ))
    print('Test 2: ', ics)
    ics = SourceSet(origin=((1, 2, 3), (4, 5, 6)), S0=1, Bx=5, on=(True, ))
    print('Test 3: ', ics)