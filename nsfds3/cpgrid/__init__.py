#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2020 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
# Creation Date : 2022-07-08 - 13:26:57
"""
-----------
DOCSTRING

-----------
"""

from ._mesh import RegularMesh
from ._cdomain import ComputationDomains
from ._geometry import Obstacle, ObstacleSet, Domain, DomainSet

__all__ = ['templates', 'RegularMesh', 'ComputationDomains',
           'ObstacleSet', 'Obstacle', 'DomainSet', 'Domain']
