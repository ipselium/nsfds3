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
# Creation Date : 2022-06-22 - 08:22:53
"""
-----------

The `templates` module provides a collection of examples to :

* build arangments of obstacles
* set curvilinear transformations for :py:class:`fdgrid.mesh.CurvilinearMesh`

-----------
"""

from nsfds3.cpgrid import Obstacle


class TestCases:
    """ Collection of test cases for obstacle arrangements. """

    def __init__(self, shape, stencil=3):
        self.shape = shape
        self.stencil = stencil

    def create_geometry(self, origins, sizes, bc=None):
        """ Obstacles separated"""
        obstacles = []
        if not bc:
            bc = ['WWWWWW', ] * len(origins)

        for origin, size, _bc in zip(origins, sizes, bc):
            obstacles.append(Obstacle(origin=origin, size=size, bc=_bc))

        return obstacles

    @property
    def case0(self):
        """ Empty domain. """
        conf = {'origins': [], 'sizes': []}
        return self.create_geometry(**conf)

    @property
    def case1(self):
        """ All possible singles... """
        conf = {'origins': [[0, 0, 0], [self.shape[0]-7, self.shape[1]-7, self.shape[2]-7], [15, 15, 15],
                            [self.shape[0]-7, 0, 0], [0, self.shape[1]-7, 0], [0, 0, self.shape[2]-7],
                            [0, self.shape[1]-7, self.shape[2]-7], [self.shape[0]-7, 0, self.shape[2]-7], [self.shape[0]-7, self.shape[1]-7, 0],
                            [0, 15, 0], [0, 0, 15], [15, 0, 0], [0, 15, 15], [15, 0, 15], [15, 15, 0],
                            [self.shape[0]-7, 15, 0], [0, 15, self.shape[2]-7], [15, self.shape[1]-7, 0],
                            [self.shape[0]-7, 0, 15], [15, 0, self.shape[2]-7], [0, self.shape[1]-7, 15],
                            [self.shape[0]-7, self.shape[1]-7, 15], [15, self.shape[1]-7, self.shape[2]-7], [self.shape[0]-7, 15, self.shape[2]-7],
                            [self.shape[0]-7, 15, 15], [15, self.shape[1]-7, 15], [15, 15, self.shape[2]-7]
                           ],
                'sizes': 27 * [[7, 7, 7], ]}
        return self.create_geometry(**conf)

    @property
    def case2(self):
        """ Two obstacles overlapped (two sides). """
        conf = {'origins': [(28, 7, 0), (8, 7, 7), (13, 10, 21)],
                'sizes': [(7, 10, 10), (15, 15, 15), (15, 15, 15)]}
        return self.create_geometry(**conf)

    @property
    def case3(self):
        """ Two obstacles overlapped (one sides). """
        conf = {'origins': [(28, 7, 0), (8, 10, 7), (13, 10, 21)],
                'sizes': [(7, 10, 10), (15, 15, 15), (15, 15, 15)]}
        return self.create_geometry(**conf)

    @property
    def case4(self):
        """ Two obstacles superimposed. """
        conf = {'origins': [(13, 10, 8), (13, 10, 22)],
                'sizes': [(10, 15, 15), (10, 15, 15)]}
        return self.create_geometry(**conf)

    @property
    def case5(self):
        """ Two obstacles superimposed. """
        conf = {'origins': [(13, 10, 8), (16, 13, 22)],
                'sizes': [(10, 15, 15), (7, 7, 15)]}
        return self.create_geometry(**conf)

    @property
    def case6(self):
        """ L arrangement. """
        conf = {'origins': [(13, 7, 8), (13, 7, 22)],
                'sizes': [(10, 10, 15), (10, 15, 15)]}
        return self.create_geometry(**conf)

    @property
    def case7(self):
        """ T arrangement/ """
        conf = {'origins': [(13, 13, 8), (13, 7, 22)],
                'sizes': [(10, 10, 15), (10, 21, 15)]}
        return self.create_geometry(**conf)

    @property
    def case8(self):
        """ Bridge arrangement. """
        conf = {'origins': [(13, 7, 8), (13, 7, 22), (13, 23, 8)],
                'sizes': [(10, 5, 15), (10, 21, 15), (10, 5, 15)]}
        return self.create_geometry(**conf)

    @property
    def case9(self):
        """ Window arrangement. """
        conf = {'origins': [(13, 7, 9), (13, 7, 22), (13, 23, 9), (13, 7, 0)],
                'sizes': [(10, 5, 14), (10, 21, 10), (10, 5, 14), (10, 21, 10)]}
        return self.create_geometry(**conf)
