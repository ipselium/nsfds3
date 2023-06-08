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

from nsfds3.cpgrid.geometry import Obstacle


class TestCases:
    """ Collection of test cases for obstacle arrangements. """

    def __init__(self, shape, stencil=11):
        self.shape = shape
        self.stencil = stencil
        self.thresh = self.stencil * 2 + 1

    def create_geometry(self, origins, sizes, bc=None):
        """ Obstacles separated"""
        obstacles = []
        if not bc:
            bc = ['W' * 2 * len(self.shape), ] * len(origins)

        for origin, size, _bc in zip(origins, sizes, bc):
            obstacles.append(Obstacle(origin=origin, size=size, env=self.shape, bc=_bc))

        return obstacles

    @property
    def all(self):
        """ Return a list of all test cases. """
        return [self.empty, self.single, self.edges,
                self.superimposed1, self.superimposed2,
                self.overlapped1, self.overlapped2,
                self.Lcell, self.Tcell, self.Ocell]

    @property
    def base(self):
        """ Base geometry. """

        shape = self.shape

        if len(self.shape) == 2:
            conf = {'origins': [(0, 80), (10, 95),                          # Two overlapped with one at bound location
                                (11, 50), (20, 65),                         # Two overlapped
                                (0, 11), (10, 24),                          # Two side to side with one at bound location
                                (46, 20), (46, 11), (70, 11),               # U (1)
                                (46, 50), (46, 41), (70, 50),               # U (2)
                                (46, 73), (56, 73), (70, 73),               # U (3)
                                (47, 113), (46, 104), (69, 113),            # U (4 - with lower edges)
                                (93, 120), (92, 124),                       # U (5 - with lower edges)
                                (92, 99), (92, 90),                         # L
                                (92, 50), (92, 60), (92, 64), (92, 74),     # F (1)
                                (92, 11), (102, 22), (102, 36),             # F (2)
                                (125, 100), (134, 90),                      # Cruz
                                (125, 70), (132, 56),                       # T
                                (128, 11), (140, 40),                       # Test
                                (shape[0] - 35, 59), (shape[0] - 35, 50), (shape[0] - 11, 50), (shape[0] - 35, 69), # O
                                (175, 0), (175, 10), (175, 20),             # I (overlapped)
                                (175, shape[1] - 40), (175, shape[1] - 30), (175, shape[1] - 20), # I (side to side)
                                ],
                    'sizes': [(15, 20), (15, 15),
                              (15, 20), (15, 15),
                              (11, 20), (11, 15),
                              (11, 11), (25, 10), (11, 20),
                              (11, 11), (35, 10), (11, 11),
                              (11, 20), (15, 11), (11, 20),
                              (11, 11), (35, 10), (11, 11),
                              (11, 5), (11, 5),
                              (11, 11), (25, 10),
                              (11, 11), (21, 5), (11, 11), (21, 5),
                              (11, 30), (11, 5), (11, 5),
                              (30, 11), (11, 25),
                              (25, 10), (11, 15),
                              (30, 30), (7, 7),
                              (11, 11), (25, 10), (11, 20), (35, 10),
                              (11, 20), (11, 20), (11, 20),
                              (11, 20), (11, 20), (11, 20)
                            ]}
        else:
            conf = {'origins': [(0, 17, 12), (14, 30, 12),
                                (11, 60, 12), (20, 75, 12),
                                (60, 60, 12), (73, 73, 12),
                                (74, 10, 12), (54, 20, 12)
                                ],
                    'sizes': [(15, 20, 15), (15, 20, 15),
                              (15, 20, 15), (15, 20, 15),
                              (15, 15, 15), (15, 15, 15),
                              (26, 40, 15), (21, 20, 15)
                              ]}
        return self.create_geometry(**conf)

    @property
    def empty(self):
        """ Empty domain. """
        conf = {'origins': [], 'sizes': []}
        return self.create_geometry(**conf)

    @property
    def single(self):
        """ Single obstacle. """

        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, ) * 2, ],
                    'sizes': [(40, ) * 2, ]}
        else:
            conf = {'origins': [(self.thresh, ) * 3, ],
                    'sizes': [(40, ) * 3, ]}
        return self.create_geometry(**conf)

    @property
    def edges(self):
        """ All possible singles... """
        height = 2 * self.thresh
        if any(s < (3 * height) + 3 * self.thresh for s in self.shape):
            raise Exception('domain too small for this test case')

        mid = [int(self.shape[i] / 2) - int(height / 2)
               for i in range(len(self.shape))]
        if len(self.shape) == 2:
            conf = {'origins': [[0, 0],
                                [mid[0], 0],
                                [0, mid[1]],
                                [mid[0], mid[1]],
                                [self.shape[0] - height, 0],
                                [0, self.shape[1] - height],
                                [mid[0], self.shape[1] - height],
                                [self.shape[0] - height, mid[1]],
                                [self.shape[0] - height, self.shape[1] - height],
                               ],
                    'sizes': 9 * [[height, height], ]}
        else:
            conf = {'origins': [[0, 0, 0],
                                [mid[0], 0, 0],
                                [0, mid[1], 0],
                                [0, 0, mid[2]],
                                [mid[0], mid[1], 0],
                                [mid[0], 0, mid[2]],
                                [0, mid[1], mid[2]],
                                [mid[0], mid[1], mid[2]],
                                [self.shape[0] - height, 0, 0],
                                [0, self.shape[1] - height, 0],
                                [0, 0, self.shape[2] - height],
                                [0, self.shape[1] - height, self.shape[2] - height],
                                [self.shape[0] - height, 0, self.shape[2] - height],
                                [self.shape[0] - height, self.shape[1] - height, 0],
                                [self.shape[0] - height, self.shape[1] - height, self.shape[2] - height],
                                [mid[0], self.shape[1] - height, 0],
                                [mid[0], 0, self.shape[2] - height],
                                [0, mid[1], self.shape[2] - height],
                                [self.shape[0] - height, mid[1], 0],
                                [0, self.shape[1] - height, mid[2]],
                                [self.shape[0] - height, 0, mid[2]],
                                [self.shape[0] - height, mid[1], mid[2]],
                                [mid[0], self.shape[1] - height, mid[2]],
                                [mid[0], mid[1], self.shape[2] - height],
                                [mid[0], self.shape[1] - height, self.shape[2] - height],
                                [self.shape[0] - height, mid[1], self.shape[2] - height],
                                [self.shape[0] - height, self.shape[1] - height, mid[2]]],
                    'sizes': 27 * [[height, height, height], ]}
        return self.create_geometry(**conf)

    @property
    def superimposed1(self):
        """ Two obstacles side to side. 
        __________
        |   ||   |
        |___||___|

        """
        height = 2 * self.thresh
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, self.thresh),
                                (self.thresh, self.thresh + height - 1)],
                    'sizes': [(height, height), (height, height)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, self.thresh),
                                (self.thresh, self.thresh, self.thresh + height - 1)],
                    'sizes': [(height, height, height),
                              (height, height, height)]}
        return self.create_geometry(**conf)

    @property
    def superimposed2(self):
        """ Two obstacles of different height side to side.
        __________
        |   ||   |
        |   ||___|
        |___|
         """
        height = 2 * self.thresh
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, self.thresh),
                                (2 * self.thresh, self.thresh + height - 1)],
                    'sizes': [(2 * height, height),
                              (height + self.thresh, height)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, self.thresh),
                                (2 * self.thresh, self.thresh, self.thresh + height - 1)],
                    'sizes': [(2 * height, 2 * height, height),
                              (height + self.thresh, height, height)]}
        return self.create_geometry(**conf)

    @property
    def Lcell(self):
        """ L arrangement. """
        height1 = 3 * self.thresh
        height2 = 2 * self.thresh
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, self.thresh),
                                (self.thresh, self.thresh + height1 - 1)],
                    'sizes': [(height1, height1),
                              (height2, height1)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, self.thresh),
                                (self.thresh, self.thresh, self.thresh + height1 - 1)],
                    'sizes': [(height2, height2, height1),
                              (height2, height1, height1)]}
        return self.create_geometry(**conf)

    @property
    def Tcell(self):
        """ T arrangement. """
        height1 = 3 * self.thresh
        height2 = 1 * self.thresh
        if len(self.shape) == 2:
            conf = {'origins': [(2 * self.thresh, self.thresh),
                                (self.thresh, self.thresh + height1 - 1)],
                    'sizes': [(height2, height1),
                              (height1, height1)]}
        else:
            conf = {'origins': [(2 * self.thresh, self.thresh, self.thresh),
                                (self.thresh, self.thresh, self.thresh + height1 - 1)],
                    'sizes': [(height2, height1, height1),
                              (height1, height1, height1)]}
        return self.create_geometry(**conf)

    @property
    def Ucell(self):
        """ Bridge arrangement. """
        height1 = (3 * self.thresh)
        height2 = (1 * self.thresh)
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, self.thresh),
                                (self.thresh, self.thresh + height2 - 1),
                                (self.thresh + 2 * height2, self.thresh + height2 - 1)],
                    'sizes': [(height1, height2),
                              (height2, height2),
                              (height2, height2)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, self.thresh),
                                (self.thresh, self.thresh, self.thresh + height2 - 1),
                                (self.thresh + 2 * height2,
                                    self.thresh, self.thresh + height2 - 1)],
                    'sizes': [(height1, height2, height2),
                              (height2, height2, height2),
                              (height2, height2, height2)]}
        return self.create_geometry(**conf)

    @property
    def Ocell(self):
        """ Window arrangement. """
        height1 = (3 * self.thresh)
        height2 = (1 * self.thresh)
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, self.thresh),
                                (self.thresh, self.thresh + height2 - 1),
                                (self.thresh + 2 * height2, self.thresh + height2 - 1),
                                (self.thresh, self.thresh + 2 * height2 - 2)],
                    'sizes': [(height1, height2),
                              (height2, height2),
                              (height2, height2),
                              (height1, height2)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, self.thresh),
                                (self.thresh, self.thresh, self.thresh + height2 - 1),
                                (self.thresh + 2 * height2, self.thresh, self.thresh + height2 - 1),
                                (self.thresh, self.thresh, self.thresh + 2 * height2 - 2)],
                    'sizes': [(height1, height2, height2),
                              (height2, height2, height2),
                              (height2, height2, height2),
                              (height1, height2, height2)]}
        return self.create_geometry(**conf)

    @property
    def overlapped1(self):
        """ Two obstacles overlapped (one sides). """
        width = 5 * self.thresh
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, 0),
                                (self.thresh + int(width / 5), width - 1)],
                    'sizes': [(width, width),
                              (width, width)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, 0),
                                (self.thresh + int(width / 5), self.thresh, width - 1)],
                    'sizes': [(width, width, width),
                              (width, width, width)]}
        return self.create_geometry(**conf)

    @property
    def overlapped2(self):
        """ Two obstacles overlapped (two sides). """
        width = 5 * self.thresh
        if len(self.shape) == 2:
            conf = {'origins': [(self.thresh, 0),
                                (self.thresh + int(width / 5), width - 1)],
                    'sizes': [(width, width),
                              (width, width)]}
        else:
            conf = {'origins': [(self.thresh, self.thresh, 0),
                                (self.thresh + int(width / 5),
                                    self.thresh + int(width / 5), width - 1)],
                    'sizes': [(width, width, width),
                              (width, width, width)]}
        return self.create_geometry(**conf)
