#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2019 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
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
#
# Creation Date : 2019-03-01 - 12:05:08
"""
-----------

Navier Stokes Finite Differences Solver

-----------
"""

import os
import time
import argparse
import pathlib
from multiprocessing import cpu_count, Pool, current_process
from nsfds3.cpgrid import build
from nsfds3.solver import CfgSetup, create_template, FDTD
from nsfds3.utils import files, headers
from nsfds3.graphics import MPLViewer


def parse_args():
    """ Parse arguments. """

    # Options gathered in some parsers
    commons = argparse.ArgumentParser(add_help=False)
    commons.add_argument('-q', '--quiet', action="store_true", help='quiet mode')
    commons.add_argument('-c', '--cfg-file', metavar='CF', dest='cfgfile',
                         help='path to config file')

    view = argparse.ArgumentParser(add_help=False)
    view.add_argument('-i', dest='nt', type=int, help='number of time iterations')
    view.add_argument('-r', dest='ref', type=int, help='reference frame for colormap')
    view.add_argument('view', nargs='*', default='p',
                      choices=['p', 'rho', 'vx', 'vz', 'vxz', 'e'],
                      help='variable(s) to plot')

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument('-d', '--dat-file', metavar='DF', dest='datafile',
                      help='path to hdf5 data file')

    time = argparse.ArgumentParser(add_help=False)
    time.add_argument('-t', '--timings', action="store_true",
                      help='Display complete timings')

    description = 'A Navier-Stokes Finite Difference Solver'
    root = argparse.ArgumentParser(prog='nsfds3', description=description)

    # Subparsers : solve/movie/show commands
    commands = root.add_subparsers(dest='command',
                                   help='see nsfds3 `command` -h for further help')

    commands.add_parser("solve", parents=[commons, view, data, time],
                        description="Navier-Stokes equation solver",
                        help="solve NS equations with given configuration")
    shw = commands.add_parser("show",
                              description="Helper commands for parameters/results analysis",
                              help="show results and simulation configuration")
    mak = commands.add_parser("make",
                              description="Make movie/sound files",
                              help="make movie/sound files")

    # show section subsubparsers : frame/probe/
    shw_cmds = shw.add_subparsers(dest='show_command',
                                  help='see -h for further help')
    shw_cmds.add_parser('frame', parents=[commons, view, data],
                        description="Extract frame from hdf5 file and display it",
                        help="show results at a given iteration")
    shw_cmds.add_parser('probes', parents=[commons, data],
                        description="Display pressure at probe locations",
                        help="plot pressure at probes locations")
    shw_cmds.add_parser('spectrogram', parents=[commons, data],
                        description="Display spectrograms at probe locations",
                        help="plot spectrograms at probes locations")
    shw_cmds.add_parser('grid', parents=[commons],
                        description="Display numerical grid mesh",
                        help="show numerical grid mesh")
    shw_cmds.add_parser('pgrid', parents=[commons],
                        description="Display physical grid mesh",
                        help="show physical grid mesh")
    shw_cmds.add_parser('domains', parents=[commons],
                        description="Display subdomains",
                        help="show domain decomposition")
    shw_cmds.add_parser('parameters', parents=[commons],
                        description="Display some simulation parameters",
                        help="display some simulation parameters")

    # make section subsubparsers : movie/wav/template
    mak_cmds = mak.add_subparsers(dest='make_command',
                                  help='see -h for further help')
    mak_cmds.add_parser("movie", parents=[commons, view, data],
                        description="Make a movie from existing results",
                        help="make movie from existing results")
    mak_cmds.add_parser("sound", parents=[commons, data],
                        description="Make sound files from existing results",
                        help="make sound files from existing results")
    mak_cmds.add_parser("template", parents=[commons, data],
                        description="Create basic configuration file",
                        help="Create basic configuration file")

    return root.parse_args()


def show(args, cfg, msh):
    """ Show simulation parameters and grid. """

    if args.show_command == 'parameters':
        headers.versions()
        headers.parameters(cfg, msh)

    elif args.show_command == 'grid':
        msh.show()

    elif args.show_command == 'domains':
        msh.show(domains=True)

    elif args.show_command == 'pgrid':
        print('Not implemented yet!')
        #msh.plot_physical(bz=cfg.show_bz, bc_profiles=cfg.bc_profiles,
        #                  probes=cfg.probes if cfg.show_probes else False)

    elif args.show_command == 'frame':
        plt = MPLViewer(cfg, msh, cfg.datafile)
        plt.show(view=args.view, iteration=cfg.nt,
                   show_bz=cfg.show_bz, show_prb=cfg.show_prb)

    elif args.show_command == 'probes':
        plt = MPLViewer(cfg, msh, cfg.datafile)
        plt.probes()

    elif args.show_command == 'spectrogram':
        plt = MPLViewer(cfg, msh, cfg.datafile)
        plt.spectrogram()

    else:
        headers.copyright()
        headers.versions()


def make(args, cfg, msh):
    """ Create a movie from a dataset. """
    if not cfg.quiet:
        headers.copyright()

    if args.make_command == 'movie':

        plt = MPLViewer(cfg, msh, cfg.datafile)
        plt.movie(view=args.view, nt=cfg.nt, ref=args.ref,
                  show_bz=cfg.show_bz, show_prb=cfg.show_prb,
                  fps=cfg.fps)

    elif args.make_command == 'sound':

        print('Not implemented yet !')
        #_ = sounds.probes_to_wave(cfg.datafile)


def template(args):
    """Make template."""
    if not args.cfgfile:
        print('Path/filename must be specified with -c option')
    else:
        cfgfile = pathlib.Path(args.cfgfile).expanduser()
        path, filename = cfgfile.parent, cfgfile.stem + cfgfile.suffix
        create_template(path=path, filename=filename)
        print(f"{cfgfile} created")


def solve(args, cfg, msh):
    """ Solve NS equations. """

    if not cfg.quiet:
        headers.copyright()

    # Simulation
    fdtd = FDTD(cfg, msh)
    fdtd.run()

    if cfg.figures:
        plt = MPLViewer(cfg, msh, cfg.datafile)
        if cfg.save_fld:
            plt.show(iteration=cfg.nt)
        if cfg.prb:
            plt.probes()


def main():
    """ Main """

    # Parse arguments
    args = parse_args()

    # Parse config file
    if args.cfgfile is not None:
        cfg = CfgSetup(args.cfgfile)
    else:
        cfg = CfgSetup()

    # Override values in config file with command line arguments
    if hasattr(args, 'quiet'):
        cfg.quiet = args.quiet if args.quiet is not None else cfg.quiet
    if hasattr(args, 'timing'):
        cfg.timings = args.timings if args.timings is not None else cfg.timings
    if hasattr(args, 'nt'):
        cfg.nt = args.nt if args.nt is not None else cfg.nt

    # Mesh arguments
    msh_args, msh_kwargs = cfg.get_mesh_config()

    # Mesh
    msh = build(*msh_args, **msh_kwargs)

    if args.command:
        globals()[args.command](args, cfg, msh)
    else:
        headers.copyright()
        print('Must specify an action among solve/make/show/loop')
        print('See nsfds3 -h for help')


if __name__ == "__main__":

    os.nice(20)
    main()
