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
Examples
--------

The `cpgrid` package contains in particular :

    * :py:func:`build_mesh`: Factory function to build a mesh grid from a given configuration
    * :py:class:`CartesianGrid`: Build cartesian grid
    * :py:class:`CurvilinearGrid`: Build curvilinear grid
    * :py:class:`Obstacle`: describes an obstacle with its faces
    * :py:class:`ObstacleSet`: Collection of Obstacle objects
    * :py:class:`TestCases`: Gather examples of obstacles arangments and curvilinear transformations
    * :py:class:`ComputationDomains`: Divide the grid into subdomains following the geometric configuration.


::

    from nsfds3.cpgrid import CartesianGrid, CurvilinearGrid, TestCases

    case = 'superimposed2'

    # 2d Cartesian Grid
    shape = 256, 256
    bc = 'AWWW'
    obstacles = getattr(TestCases, case)(shape)

    mesh2d = CartesianGrid(shape=shape, bc=bc, obstacles=obstacles)
    mesh2d.show(buffer=True, domains=False, 
                kwargs_obstacles=dict(hatch='/', facecolor='r'),
                kwargs_buffer=dict(hatch='/', fill=True, facecolor='b', alpha=0.1)
                )


    # 2d Curvilinear Grid
    def func(x, y):
        v = y * (1 + 1000*x**2 + 0.02 * np.sin(4 * np.pi * x / x.min()))
        return x.copy(), v


    shape = 256, 256
    origin = 128, 128
    steps = 1e-4, 1e-4
    bc = 'PPWW'
    obstacles = getattr(TestCases, case)(shape)

    mesh2d = CurvilinearGrid(shape=shape, origin=origin, steps=steps, bc=bc, 
                            obstacles=obstacles, curvilinear_func=func)
    mesh2d.show(N=2, kwargs_obstacles=dict(hatch='/', facecolor='r'))


    # 3d Cartesian Grid
    shape = 256, 240, 220
    bc = 'AWWAWW'
    obstacles = getattr(TestCases, case)(shape)

    mesh3d = CartesianGrid(shape=shape, bc=bc, obstacles=obstacles)

    slices = [o + int(3*s/4) for o, s in zip(mesh3d.obstacles[0].origin, mesh3d.obstacles[0].size)][::-1]

    mesh3d.show(buffer=True, domains=False, slices=slices,
                kwargs_obstacles=dict(hatch='/', facecolor='r', alpha=0.7))


    # 3d Curvilinear Grid
    def func(x, y, z):
        u = x.copy()
        v = y * (1 + 0.01 * np.sin(4 * np.pi * x / x.min())) - 0.0005 * np.sin(0.5 * np.pi * (x - x.min())/ x.min())
        w = z.copy() * (1 + 0.005 *  np.sin(4 * np.pi * x / x.min()))
        return u, v, w


    shape = 256, 240, 220
    steps = 1e-4, 1e-4, 1e-4
    bc = 'AWWAWW'
    obstacles = getattr(TestCases, case)(shape)

    mesh3d = CurvilinearGrid(shape=shape, steps=steps, bc=bc, 
                            obstacles=obstacles, curvilinear_func=func)

    slices = [o + int(3*s/4) for o, s in zip(mesh3d.obstacles[0].origin, mesh3d.obstacles[0].size)][::-1]

    mesh3d.show(buffer=True, domains=False, grid=True, slices=slices,
                kwargs_obstacles=dict(hatch='/', facecolor='r', alpha=0.7))
"""

from nsfds3.cpgrid.mesh import build_mesh, CartesianGrid, CurvilinearGrid
from nsfds3.cpgrid.cdomain import ComputationDomains
from nsfds3.cpgrid.geometry import Obstacle, ObstacleSet
from nsfds3.cpgrid.templates import TestCases

__all__ = ['build_mesh', 'TestCases', 'CartesianGrid', 'CurvilinearGrid',
           'ComputationDomains', 'ObstacleSet', 'Obstacle']
