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
# Creation Date : 2022-06-23 - 17:36:20
"""
-----------

nsfds3 headers !

-----------
"""

from datetime import datetime as _dt
import platform as _pf
import sys as _sys
import numpy as _np
import plotly as _ply
import matplotlib as _plt
import rich.box as _box
from rich.table import Table as _Table
from rich.panel import Panel as _Panel
from rich.text import Text as _Text
from rich.console import Console

console = Console()


def copyright():
    title = f"[magenta]nsfds2 v0.1.0 -- Copyright (C) 2016-{_dt.now().year} -- Cyril Desjouy\n"
    cp = "This program comes with [u]ABSOLUTELY NO WARRANTY[/u]. " + \
         "This is free software, and you are welcome to redistribute it " + \
         "[blue center]under certain conditions; See GNU GPL v3 for more details."
    cp = _Text("\nThis program comes with ", justify="center", style='magenta italic')
    cp.append("ABSOLUTELY NO WARRANTY ", style='italic bold underline')
    cp.append("This is free software, and you are welcome to redistribute it " +
              "under certain conditions; See GNU GPL v3 for more details.", style='italic')

    panel = _Panel(cp, box=_box.DOUBLE, title=title, expand=True, style='blue')
    console.print(panel)


def versions():
    """ Displays versions of the dependencies. """
    table = _Table(box=_box.ROUNDED, expand=True)
    table.add_column('software', style='bold blue', justify='center')
    table.add_column('version', style='italic blue', justify='center')
    table.add_row('system', _pf.platform())
    table.add_row('python', _sys.version.split(' ', maxsplit=1)[0])
    table.add_row('numpy', _np.__version__)
    table.add_row('plotly', _ply.__version__)
    table.add_row('maplotlib', _plt.__version__)
    console.print(table)


if __name__ == '__main__':

    copyright()
    versions()