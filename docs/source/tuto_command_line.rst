============
Command line
============

Basics
======

**nsfds3** can be used from a terminal with::

   nsfds3 solve|make|show

* *solve* : solves Navier-Stokes equation using a default configuration
* *make* : makes movie or sound files from results obtained with *solve* subcommand
* *show* : set of commands to display simulations parameters, grid or to inspect simulation results

See `-h` option for further help::

   nsfds3 solve -h
   nsfds3 make -h 		# 'movie' and 'sound' subcommands
   nsfds3 show -h 		# 'parameters', 'grid', 'frame', 'probes', 'spectrogram' subcommands

The solve subcommand
====================

The `solve` subcommand launch the solver. By default, the solver uses the 
last configuration used (or the default configuration from 
`nsfds3.solver.CfgSetup()` if it does not found any last configuration). 
To target a configuration file, use the `-c` argument as follows::

   nsfds3 solve -c my_config_file.conf

To see other options::

   nsfds3 solver -h

The make subcommand
===================

The `make` subcommand can either generate movie (.mp4 file) or sound (.wav file)
from an hdf5 file::

   nsfds3 make sound
   nsfds3 make movie

Work with sounds
----------------

Sounds are generated from the probes (pressure) saved in the data file 
(hdf5 file). If no probe has been set in the computation domain, no sound 
will be generated from `nsfds3 make sound`. 

Work with movies
----------------

Movies can be created from an hdf5 file (or using the .cfg file of the 
simulation) if the `save` option has been selected.
Then, the following variables are allowed as argument of the movie subcommand:

+------+--------------------------------------------+
| var  | variable                                   |
+======+============================================+
| p    | pressure                                   |
+------+--------------------------------------------+
| rho  | density                                    |
+------+--------------------------------------------+
| vx   | x component of the velocity                |
+------+--------------------------------------------+
| vy   | y component of the velocity                |
+------+--------------------------------------------+
| vz   | z component of the velocity (only for 3d)  |
+------+--------------------------------------------+
| e    | energy                                     |
+------+--------------------------------------------+
| wz   | z component of the vorticity               |
+------+--------------------------------------------+
| wy   | y component of the vorticity (only for 3d) |
+------+--------------------------------------------+
| wx   | x component of the vorticity (only for 3d) |
+------+--------------------------------------------+

For instance, to create a movie of the x-component of the velocity, one can use::

   nsfds3 make movie vx -c my_confif_file.conf

or directly from a target data file::

   nsfds3 make movie rho -d my_data_file.hdf5


The show subcommand
===================

The `show` subcommand provides a set of commands to show simulation parameters,
results, or grid configuration. The main `show` subcommands are:

:frame:  Display an acoustic variable at a given iteration (works like `movie`)
:probes: Display pressure field(s) as a function of time at probe location(s)
:spectrogram: Display spectrogram(s) at probe location(s)
:parameters: Display some simulation parameters
:grid: Display the grid
:domains: Display the computation domains (only is free=False)

For instance, to display the density at iteration 100::

   nsfds3 show frame rho -n 100 -d my_data_file.hdf5