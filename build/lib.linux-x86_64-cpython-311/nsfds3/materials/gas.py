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
# Creation Date : 2023-07-21 - 00:33:00
"""
Thermophysical parameters.
"""

import numpy as np
from scipy import constants


class Air:
    """ Thermophysical parameters of Air.

    Parameters
    ----------
    rho0: float, optional
        Density in kg.m^{-3}
    T0: float, optional
        Temperature in C
    gamma: float, optional
        Ratio of heat capacities
    norm: bool, optional
        If True, normalize the thermophysical parameters
    """

    Tref = constants.zero_Celsius  # Temperature 0°C [K]
    Rgp = constants.R              # Molar gas constant
    Ssu = 110.4                    # Sutherland constant
    mu0 = 17.16e-6                 # Dynamic viscosity at T = 0 deg [Pa.s]
    k = 0.026                      # thermal conductivity. Almost constant % temperature.
    Mmol = 28.9647e-3              # Molar mass of air [kg/mol]

    def __init__(self, rho0=1.2, T0=20, gamma=1.4, norm=False):

        self._rho0 = rho0
        self.T0 = T0 + self.Tref
        self.gamma = gamma

        self._norm = False
        self._norm_factor = self.T0 / self.c0   # [K.m^{-1}.s]
        if norm:
            self.norm = True

    @property
    def norm(self):
        """ Report whether or not the parameters are normalized. """
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = bool(value)

    @property
    def rho0(self):
        """ Density [kg.m^{-3}]. """
        if self.norm:
            return 1.
        return self._rho0

    @property
    def c0(self):
        """ Celerity of sound [m.s^{-1}]. """
        if self.norm:
            return 1.
        return np.sqrt(self.T0 * self.R * self.gamma)

    @property
    def p0(self):
        """ Static pressure [Pa]. """
        return self.rho0 * self.c0**2 / self.gamma

    @property
    def R(self):
        """ Gas constant of air [J.kg^{-1}.K^{-1}]. """
        if self.norm:
            return self.Rgp / self.Mmol * self._norm_factor
        return self.Rgp / self.Mmol

    @property
    def Cv(self):
        """ Mass heat capacity at constant volume [J.kg^{-1}.K^{-1}]. """
        return self.R / (self.gamma - 1)

    @property
    def Cp(self):
        """ Mass heat capacity at constant pressure [J.kg^{-1}.K^{-1}]. """
        return self.Cv * self.gamma

    @property
    def mu(self):
        """ Dynamic viscosity at T0 [Pa.s]. """
        return (self.mu0 * (self.T0 / self.Tref)**(3. / 2.) *
                (self.Tref + self.Ssu) / (self.T0 + self.Ssu))

    @property
    def prandtl(self):
        """ Prandtl number. """
        return self.Cp * self.mu / self.k

    def __str__(self):
        s = f'- Static pressure                 : {self.p0} Pa\n'
        s += f'- Density                         : {self.rho0} kg.m^{-3}\n'
        s += f'- Temperature                     : {self.T0} K ({self.T0 - self.Tref}°C)\n'
        s += f'- Celerity                        : {self.c0} m/s\n'
        s += f'- Heat capacity ratio             : {self.gamma}\n'
        s += f'- Isochoric mass heat capacity    : {self.Cv} J.kg^{-1}.K^{-1}\n'
        s += f'- Isobaric mass Heat capacity (P) : {self.Cp} J.kg^{-1}.K^{-1}\n'
        s += f'- Prandtl number                  : {self.prandtl}\n'
        s += f'- Dynamic viscosity               : {self.mu} Pa.s\n'
        return s

    def __repr__(self):
        return self.__str__()
