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
The `templates` module provides a collection of examples to:

* build arangments of obstacles
* set curvilinear transformations

This module contains:

* :py:func:`create_geometry` : returns a list of obstacles from their origin/size/bc parameters
* :py:class:`TestCases` : Gather differents arangments of obstacles
"""

import numpy as _np
from types import FunctionType
from nsfds3.cpgrid.geometry import Obstacle


def create_geometry(shape, origins, sizes, bc=None):
    """ Obstacles separated"""
    obstacles = []
    if not bc:
        bc = ['W' * 2 * len(shape), ] * len(origins)

    for origin, size, _bc in zip(origins, sizes, bc):
        obstacles.append(Obstacle(origin=origin, size=size, env=shape, bc=_bc))

    return obstacles


class TestCases:
    """
    Collection of obstacles arangments and curvilinear transformations.
    """

    @classmethod
    @property
    def all(cls):
        methods = dir(cls)
        methods.remove('all')
        return [getattr(cls, m) for m in methods
                if not m.startswith('_') and not m.startswith('curv')
                and not m.startswith('evolution') and
                type(getattr(cls, m)) == FunctionType]

    @staticmethod
    def empty(shape, stencil=11):
        """ Empty domain. """
        conf = {'origins': [], 'sizes': []}
        return create_geometry(shape, **conf)

    @staticmethod
    def single(shape, stencil=11):
        """ Single obstacle. """

        thresh = stencil * 2 + 1

        if len(shape) == 2:
            conf = {'origins': [(thresh, ) * 2, ],
                    'sizes': [(40, ) * 2, ]}
        else:
            conf = {'origins': [(thresh, ) * 3, ],
                    'sizes': [(40, ) * 3, ]}
        return create_geometry(shape, **conf)

    @staticmethod
    def evolution_sine(time):
        """ Sinusoïdal time evolution """
        f = 1 / (50 * time[1])
        amp = 1
        return amp * _np.sin(2 * _np.pi * f * time)

    @staticmethod
    def single_source(shape, stencil=11):
        """ Single obstacle with wall source.

        ::

            _____
            |   |
            |___|

        """

        thresh = stencil * 2 + 1
        obs = Obstacle(origin=(thresh, ) * len(shape), size=(40, ) * len(shape), env=shape, bc='WVWW' + 2 * (len(shape) - 2) * 'W')
        obs.face_right.set_source(TestCases.evolution_sine, 'sine')
        return [obs, ]

    @staticmethod
    def base(shape, stencil=11):
        """ Base geometry. """

        if len(shape) == 2:
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
        return create_geometry(shape, **conf)

    @staticmethod
    def edges(shape, stencil=11):
        """ All possible single edges... """

        thresh = stencil * 2 + 1
        height = 2 * thresh

        if any(s < (3 * height) + 3 * thresh for s in shape):
            raise Exception('domain too small for this test case')

        mid = [int(shape[i] / 2) - int(height / 2)
               for i in range(len(shape))]

        if len(shape) == 2:
            conf = {'origins': [[0, 0],
                                [mid[0], 0],
                                [0, mid[1]],
                                [mid[0], mid[1]],
                                [shape[0] - height, 0],
                                [0, shape[1] - height],
                                [mid[0], shape[1] - height],
                                [shape[0] - height, mid[1]],
                                [shape[0] - height, shape[1] - height],
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
                                [shape[0] - height, 0, 0],
                                [0, shape[1] - height, 0],
                                [0, 0, shape[2] - height],
                                [0, shape[1] - height, shape[2] - height],
                                [shape[0] - height, 0, shape[2] - height],
                                [shape[0] - height, shape[1] - height, 0],
                                [shape[0] - height, shape[1] - height, shape[2] - height],
                                [mid[0], shape[1] - height, 0],
                                [mid[0], 0, shape[2] - height],
                                [0, mid[1], shape[2] - height],
                                [shape[0] - height, mid[1], 0],
                                [0, shape[1] - height, mid[2]],
                                [shape[0] - height, 0, mid[2]],
                                [shape[0] - height, mid[1], mid[2]],
                                [mid[0], shape[1] - height, mid[2]],
                                [mid[0], mid[1], shape[2] - height],
                                [mid[0], shape[1] - height, shape[2] - height],
                                [shape[0] - height, mid[1], shape[2] - height],
                                [shape[0] - height, shape[1] - height, mid[2]]],
                    'sizes': 27 * [[height, height, height], ]}
        return create_geometry(shape, **conf)

    @staticmethod
    def superimposed1(shape, stencil=11):
        """ Two obstacles side to side.

        ::

            __________
            |   ||   |
            |___||___|

        """

        thresh = stencil * 2 + 1
        height = 2 * thresh

        if len(shape) == 2:
            conf = {'origins': [(thresh, thresh),
                                (thresh, thresh + height - 1)],
                    'sizes': [(height, height), (height, height)]}
        else:
            conf = {'origins': [(thresh, thresh, thresh),
                                (thresh, thresh, thresh + height - 1)],
                    'sizes': [(height, height, height),
                              (height, height, height)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def superimposed2(shape, stencil=11):
        """ Two obstacles of different height side to side.

        ::

            __________
            |   ||   |
            |   ||___|
            |___|

        """

        thresh = stencil * 2 + 1
        height = 2 * thresh

        if len(shape) == 2:
            conf = {'origins': [(thresh, thresh),
                                (2 * thresh, thresh + height - 1)],
                    'sizes': [(2 * height, height),
                              (height + thresh, height)]}
        else:
            conf = {'origins': [(thresh, thresh, thresh),
                                (2 * thresh, thresh, thresh + height - 1)],
                    'sizes': [(2 * height, 2 * height, height),
                              (height + thresh, height, height)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def Lcell(shape, stencil=11):
        """ L arrangement.

        ::

            _____
            |   |
            |   |_____
            |___||___|

        """

        thresh = stencil * 2 + 1
        height1 = 3 * thresh
        height2 = 2 * thresh

        if len(shape) == 2:
            conf = {'origins': [(thresh, thresh),
                                (thresh, thresh + height1 - 1)],
                    'sizes': [(height1, height1),
                              (height2, height1)]}
        else:
            conf = {'origins': [(thresh, thresh, thresh),
                                (thresh, thresh, thresh + height1 - 1)],
                    'sizes': [(height2, height2, height1),
                              (height2, height1, height1)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def Tcell(shape, stencil=11):
        """ T arrangement.

        ::

          _________
          |_______|
            |   |
            |___|

        """

        thresh = stencil * 2 + 1
        height1 = 3 * thresh
        height2 = 1 * thresh

        if len(shape) == 2:
            conf = {'origins': [(2 * thresh, thresh),
                                (thresh, thresh + height1 - 1)],
                    'sizes': [(height2, height1),
                              (height1, height1)]}
        else:
            conf = {'origins': [(2 * thresh, thresh, thresh),
                                (thresh, thresh, thresh + height1 - 1)],
                    'sizes': [(height2, height1, height1),
                              (height1, height1, height1)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def Ucell(shape, stencil=11):
        """ Bridge arrangement.

        ::

            _____     _____
            |   |     |   |
            |   |_____|   |
            |___||___||___|

        """

        thresh = stencil * 2 + 1
        height1 = (3 * thresh)
        height2 = (1 * thresh)

        if len(shape) == 2:
            conf = {'origins': [(thresh, thresh),
                                (thresh, thresh + height2 - 1),
                                (thresh + 2 * height2, thresh + height2 - 1)],
                    'sizes': [(height1, height2),
                              (height2, height2),
                              (height2, height2)]}
        else:
            conf = {'origins': [(thresh, thresh, thresh),
                                (thresh, thresh, thresh + height2 - 1),
                                (thresh + 2 * height2,
                                    thresh, thresh + height2 - 1)],
                    'sizes': [(height1, height2, height2),
                              (height2, height2, height2),
                              (height2, height2, height2)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def Ocell(shape, stencil=11):
        """ Window arrangement.

        ::

            _______________
            |   ||___||   |
            |   |     |   |
            |   |_____|   |
            |___||___||___|

        """

        thresh = stencil * 2 + 1
        height1 = 3 * thresh
        height2 = 1 * thresh

        if len(shape) == 2:
            conf = {'origins': [(thresh, thresh),
                                (thresh, thresh + height2 - 1),
                                (thresh + 2 * height2, thresh + height2 - 1),
                                (thresh, thresh + 2 * height2 - 2)],
                    'sizes': [(height1, height2),
                              (height2, height2),
                              (height2, height2),
                              (height1, height2)]}
        else:
            conf = {'origins': [(thresh, thresh, thresh),
                                (thresh, thresh, thresh + height2 - 1),
                                (thresh + 2 * height2, thresh, thresh + height2 - 1),
                                (thresh, thresh, thresh + 2 * height2 - 2)],
                    'sizes': [(height1, height2, height2),
                              (height2, height2, height2),
                              (height2, height2, height2),
                              (height1, height2, height2)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def overlapped1(shape, stencil=11):
        """ Two obstacles overlapped (one sides). """

        thresh = stencil * 2 + 1
        width = 5 * thresh

        if len(shape) == 2:
            conf = {'origins': [(thresh, 0),
                                (thresh + int(width / 5), width - 1)],
                    'sizes': [(width, width),
                              (width, width)]}
        else:
            conf = {'origins': [(thresh, thresh, 0),
                                (thresh + int(width / 5), thresh, width - 1)],
                    'sizes': [(width, width, width),
                              (width, width, width)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def overlapped2(shape, stencil=11):
        """ Two obstacles overlapped (two sides). """

        thresh = stencil * 2 + 1
        width = 5 * thresh

        if len(shape) == 2:
            conf = {'origins': [(thresh, 0),
                                (thresh + int(width / 5), width - 1)],
                    'sizes': [(width, width),
                              (width, width)]}
        else:
            conf = {'origins': [(thresh, thresh, 0),
                                (thresh + int(width / 5),
                                    thresh + int(width / 5), width - 1)],
                    'sizes': [(width, width, width),
                              (width, width, width)]}
        return create_geometry(shape, **conf)

    @staticmethod
    def curv_mountain(x, y):
        """ Curvilinear function example. """
        xsine = _np.linspace(-_np.pi, _np.pi, x.shape[0])
        sine = _np.sin(xsine/0.1)
        profile = _np.zeros_like(x)
        for i in range(x.shape[1]):
            profile[:, i] = (2/(i/50+1)) * (sine - xsine**2)

        return x.copy(), y + profile