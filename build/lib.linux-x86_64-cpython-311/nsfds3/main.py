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
---

Navier Stokes Finite Differences Solver

---
"""

import os
import argparse
from rich import print
from nsfds3.cpgrid import build_mesh
from nsfds3.solver import CfgSetup, FDTD
from nsfds3.utils import headers, probes_to_wavfile
from nsfds3.graphics import MPLViewer


def parse_args():
    """ Parse arguments. """

    # Options gathered in some parsers
    commons = argparse.ArgumentParser(add_help=False)
    commons.add_argument('-q', '--quiet', action='store_true',
                         help='Run nsfds3 in quiet mode')
    commons.add_argument('-l', '--notlast', action='store_true',
                         help='Forbid the use of last configuration')
    commons.add_argument('-f', '--force-build', dest='force', action='store_true',
                         help='Force grid building')
    commons.add_argument('-c', '--cfgfile', 
                         help='Path to .conf file to use')

    view = argparse.ArgumentParser(add_help=False)
    view.add_argument('-n', '--nt', type=int,
                      help='Number of time iterations')
    view.add_argument('-r', '--ref', type=int,
                      help='Reference frame for colormap')
    view.add_argument('view', nargs='*', default='p',
                      choices=['p', 'rho', 'vx', 'vz', 'wx', 'wy', 'wz', 'e'],
                      help='Variable to plot')

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument('-d', '--datapath',
                      help='Path to hdf5 data file to use')

    time = argparse.ArgumentParser(add_help=False)
    time.add_argument('-t', '--timings', action="store_true",
                      help='Display complete timings')

    description = 'A 3d Navier-Stokes Finite Difference Solver'
    root = argparse.ArgumentParser(prog='nsfds3', description=description)

    # Subparsers : solve/movie/show commands
    commands = root.add_subparsers(dest='command',
                                   help='See nsfds3 `command` -h for further help')

    slv = commands.add_parser("solve", parents=[commons, view, data, time],
                              description="Navier-Stokes equation solver",
                              help="Solve NS equations with given configuration")
    shw = commands.add_parser("show", parents=[commons, ],
                              description="Helper commands for parameters/results inspection",
                              help="Show results and simulation configuration")
    mak = commands.add_parser("make", parents=[commons, ],
                              description="Make movie/sound files",
                              help="Make movie/sound files")

    # show section subsubparsers : frame/probe/
    shw_cmds = shw.add_subparsers(dest='show_command',
                                  help='See -h for further help')
    shw_cmds.add_parser('frame', parents=[commons, view, data],
                        description="Extract frame from hdf5 file and display it",
                        help="Show results at a given iteration")
    shw_cmds.add_parser('probes', parents=[commons, data],
                        description="Display pressure at probe locations",
                        help="Plot pressure at probes locations")
    shw_cmds.add_parser('spectrogram', parents=[commons, data],
                        description="Display spectrograms at probe locations",
                        help="Plot spectrograms at probes locations")
    shw_cmds.add_parser('grid', parents=[commons],
                        description="Display numerical grid mesh",
                        help="Show numerical grid mesh")
    shw_cmds.add_parser('domains', parents=[commons],
                        description="Display subdomains",
                        help="Show domain decomposition")
    shw_cmds.add_parser('parameters', parents=[commons],
                        description="Display some simulation parameters",
                        help="Display some simulation parameters")

    # make section subsubparsers : movie/wav
    mak_cmds = mak.add_subparsers(dest='make_command',
                                  help='See -h for further help')
    mak_cmds.add_parser("movie", parents=[commons, view, data],
                        description="Make a movie file from existing results",
                        help="Make a movie file from existing results")
    mak_cmds.add_parser("sound", parents=[commons, data],
                        description="Make sound files from existing results",
                        help="Make sound files from existing results")

    return root.parse_args()


def show(args, cfg, msh):
    """ Show simulation parameters and grid. """

    if args.show_command == 'parameters':
        headers.versions()
        print(cfg)

    elif args.show_command == 'grid':
        msh.show()

    elif args.show_command == 'domains':
        msh._computation_domains.show(domains=True)

    elif args.show_command == 'frame':
        plt = MPLViewer(cfg, msh, cfg.datapath)
        plt.show(view=args.view, iteration=cfg.nt,
                   buffer=cfg.show_bz, probes=cfg.show_prb)

    elif args.show_command == 'probes':
        plt = MPLViewer(cfg, msh, cfg.datapath)
        plt.probes()

    elif args.show_command == 'spectrogram':
        plt = MPLViewer(cfg, msh, cfg.datapath)
        plt.spectrogram()

    else:
        headers.versions()


def make(args, cfg, msh):
    """ Create a movie from a dataset. """

    if args.make_command == 'movie':

        plt = MPLViewer(cfg, msh, cfg.datapath)
        plt.movie(view=args.view, nt=cfg.nt, ref=args.ref,
                  buffer=cfg.show_bz, probes=cfg.show_prb,
                  fps=cfg.fps)

    elif args.make_command == 'sound':
        _ = probes_to_wavfile(cfg.datapath)


def solve(args, cfg, msh):
    """ Solve NS equations. """

    # Simulation
    fdtd = FDTD(cfg, msh)
    fdtd.run()

    if cfg.show_fig:
        plt = MPLViewer(cfg, msh, cfg.datapath)
        if cfg.save_fld:
            plt.show(iteration=cfg.nt)
        if cfg.prb:
            plt.probes()


def main():
    """ Main function for nsfd3 command line api. """

    # Headers
    headers.copyright()

    # Parse arguments
    args = parse_args()

    # Parse config file
    cfg = CfgSetup(args.cfgfile, last=not args.notlast)

    # Override values in config file with command line arguments
    if hasattr(args, 'quiet'):
        cfg.quiet = args.quiet if args.quiet is not None else cfg.quiet
    if hasattr(args, 'timing'):
        cfg.timings = args.timings if args.timings is not None else cfg.timings
    if hasattr(args, 'nt'):
        cfg.nt = args.nt if args.nt is not None else cfg.nt

    # legacy : nsfds2 allowed for multiple views on the same frame, not nsfds3!
    if hasattr(args, 'view'):
        if isinstance(args.view, list):
            args.view = args.view[0]



    if args.command:
        msh = None
        if not args.force:
            msh = cfg.get_grid_backup()
            if msh is not None and not cfg.quiet:
                print(f'Got [bold red]existing {type(msh).__name__}[/] for this configuration. Skip grid generation...')

        if args.force or msh is None:
            msh = build_mesh(cfg)

        globals()[args.command](args, cfg, msh)
    else:
        print('Must specify an action among solve/make/show/loop')
        print('See nsfds3 -h for help')


if __name__ == "__main__":

    os.nice(20)
    main()
