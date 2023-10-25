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
# Creation Date : 2023-07-20 - 13:56:45
"""
-----------
DOCSTRING

-----------
"""


from nsfds3.cpgrid import Obstacle


def blabla(shape):

    if len(shape) == 3:
        o1 = Obstacle(origin=(8, 8, 8), size=(9, 9, 9), env=shape, bc="WWWWWW")
    else:
        o1 = Obstacle(origin=(8, 8), size=(9, 9), env=shape, bc="WWWW")

    return [o1, ]

