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
# Creation Date : 2022-07-08 - 13:27:06
"""
The `graphics` package provides some helper classes and functions to represent meshes graphically.

    * :py:class:`nsfds3.graphics.MeshViewer`: Graphical tool to visualize Mesh objects graphically.
    * :py:class:`nsfds3.graphics.CPViewer`: MeshViewer specialization adapted to ComputationDomains.
    * :py:class:`nsfds3.graphics.MPLViewer`: MeshViewer specialization adapted to libfds.Fields or hdf5 files.

The main viewer used in nsfds3 is :py:class:`nsfds3.graphics.MPLViewer`.

Example
-------

::

    from nsfds3.graphics import MPLViewer
    from nsfds3.utils import get_objects

    path = 'data/'
    filename = 'reference'
    cfg, msh = get_objects(path, filename)
    viewer = MPLViewer(cfg, msh, cfg.datapath')

    viewer2d.movie(view='p', buffer=False, probes=True)  # Make movie
    viewer2d.probes()                                    # figure with probe evolutions
    viewer2d.spectrogram()                               # figure with spectrogram
"""

from nsfds3.graphics.graphics import MPLViewer, MeshViewer, CPViewer

__all__ = ['MPLViewer', 'MeshViewer', 'CPViewer']