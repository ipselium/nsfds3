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
# Creation Date : 2022-06-09 - 22:15:26
"""
-----------

Some helper classes and functions to represent meshes graphically.

-----------
"""

import copy
import collections.abc
import matplotlib
import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm


def cmap_exists(name):
    """ Check if the cmap named name is already registered. """
    try:
        matplotlib.cm.get_cmap(name)
    except ValueError:
        return False
    return True


def cmap_mask(stencil=11):
    """ Discrete cmap with 4 colours to describe masks."""
    nodes = [-stencil, 0, 1, stencil]
    colors = ["mistyrose", "black", "white", "paleturquoise"]
    return ListedColormap(colors), BoundaryNorm(nodes, len(colors) - 1)


def cmap_jet():
    """
    Custom jet colormap

    Note
    ----
    howto : http://matplotlib.org/examples/pylab_examples/custom_cmap.html
    """

    if not cmap_exists('mjet'):
        cdictjet = {'blue': ((0.0, 1., 1.),
                             (0.11, 1, 1),
                             (0.34, 1, 1),
                             (0.48, 1, 1),
                             (0.52, 1, 1),
                             (0.65, 0, 0),
                             (1, 0, 0)),
                    'green': ((0.0, 0.6, 0.6),
                              (0.125, 0.8, 0.8),
                              (0.375, 1, 1),
                              (0.48, 1, 1),
                              (0.52, 1, 1),
                              (0.64, 1, 1),
                              (0.91, 0, 0),
                              (1, 0, 0)),
                    'red': ((0.0, 0, 0),
                            (0.35, 0, 0),
                            (0.48, 1, 1),
                            (0.52, 1, 1),
                            (0.66, 1, 1),
                            (0.8, 1, 1),
                            (1, 0., 0.))
                    }
        cmc = LinearSegmentedColormap('mjet', cdictjet, 1024)
        _plt.register_cmap(name='mjet', cmap=cmc)
    else:
        cmc = matplotlib.cm.get_cmap('mjet')

    return cmc


def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = _plt.cm.get_cmap(cmap)
    colors = cmap(_np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = _np.sqrt(_np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, _np.newaxis]

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


class MidPointNorm(matplotlib.colors.Normalize):
    """ Adjust cmap using a midpoint as reference.

    Note
    ----
    From https://stackoverflow.com/questions/
    7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):

        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):

        if self.vmin == 0 and self.vmax == 0:
            self.vmin, self.vmax = -1, 1

        elif self.vmin >= self.vmax:
            self.vmin, self.vmax = self.vmin, self.vmin + self.vmin if self.vmin > 0 else 0

        else:
            if self.midpoint != self.vmax:
                vmin = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) /
                                              (self.midpoint - self.vmax))))
            else:
                vmin = -1

            if self.midpoint != self.vmin:
                vmax = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) /
                                            (self.midpoint - self.vmin))))
            else:
                vmax = 1

        mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [vmin, mid, vmax]
        return _np.ma.masked_array(_np.interp(value, x, y))


def extend_range(a, b, percent=5):
    """ Extends the numerical range (a, b) by percent. """
    value = (b - a) * percent / 100
    return a - value, b + value


def dict_update(d1, d2):
    out1 = copy.deepcopy(d1)
    out2 = copy.deepcopy(d2)
    if all((isinstance(d, collections.abc.Mapping) for d in (out1, out2))):
        for k, v in out2.items():
            out1[k] = dict_update(out1.get(k), v)
        return out1
    return out2


def fig_scale(ax1, ax2, ref=None):
    """ Return ideal size ratio. """
    if isinstance(ax1, (tuple, list)):
        s1 = sum(ax.max() - ax.min() for ax in ax1)
    else:
        s1 = ax1.max() - ax1.min()

    if isinstance(ax2, (tuple, list)):
        s2 = sum(ax.max() - ax.min() for ax in ax2)
    else:
        s2 = ax2.max() - ax2.min()

    ratio = min(s1, s2) / max(s1, s2)
    b1 = 1 / (1 + ratio)
    b2 = 1 - b1

    if ref:
        if s1 < s2:
            b2, b1 = ref, ref * b1 / b2
        else:
            b1, b2 = ref, ref * b2 / b1

    return b2 if s1 < s2 else b1, b2 if s2 < s1 else b1