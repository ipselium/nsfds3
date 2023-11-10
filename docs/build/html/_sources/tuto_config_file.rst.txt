==================
Configuration file
==================

Introduction
============

**nsfds3** usually use a configuration file to setup all simulation parameters. 
This file can contain the following entries.

::

   [configuration]
   version = 0.1.0
   data dir = results/               # path to data file
   data file = tmp                   # data filename
   timings = True|False              # Display timing detail
   quiet = True|False                # Quiet mode
   cpu = 1                           # Number of cpu used by the solver
   free = True|False                 # Free memory (ComputationDomain intermediate vars)
   comp = None|lzf                   # Compression of data files

   [thermophysic]
   norm = True|False                 # Normalize thermophysical values.
   rho0 = 1.2                        # Density of fluid (kg/m3)
   t0 = 20.0                         # Ambiant temperature (°C)
   gamma = 1.4                       # Heat capacity ratio

   [geometry]
   geofile = None|path               # Path to .py file (for geoname/curvname and other functions)
   geoname = None|func_name          # Function name for geometry
   curvname = None|func_name         # Function name for curvilinear coordinates
   bc = AWPP                         # Boundary conditions. Must be a mix of AWP
   shape = (100, 80)                 # Grid points. Can be 2 or 3 elements tuple
   origin = None|(10, 10)            # Origin of the grid. Can be 2 or 3 elements tuple
   steps = None|(1, 1)               # spatial steps. Can be 2 or 3 elements tuple
   flat = None|(ax, idx)             # Make flat a 3d geometry along ax at idx location
   bz grid points = 20               # Number of points of the buffer zone (BZ)
   bz filter order = 3               # Order of the filtering in the BZ
   bz stretch order = 3              # Order of the grid stretching in the BZ
   bz stretch factor = 2             # Factor of the grid stretching in the BZ

   [initial pulses]
   on = True|False                   # Whether or not to setup initial pulses
   origins = (),                     # Origins of the pulses (tuple of tuples)
   amplitudes = ()                   # Amplitudes of the pulses
   widths = ()                       # widths of the pulses

   [sources]
   on = True|False                   # Whether or not to setup sources
   origins = (),                     # Origins of the sources (tuple of tuples)
   amplitudes = ()                   # Amplitudes of the sources
   widths = ()                       # widths of the sources
   evolutions = ()                   # time evolution of the sources

   [flow]
   type = None                       # Flow type (not supported for now)
   components = (0, 0)               # Components of the flow velocity [m/s]

   [solver]
   resume = True|False               # Whether or not to resume a computation
   nt = 500                          # Number of time iterations
   ns = 10                           # Field saving frequency
   cfl = 0.5                         # Courant–Friedrichs–Lewy number
   probes = ()                       # Probe locations. Must be tuple of tuples
   save fields = True|False          # Whether or not to save fields
   viscous fluxes = True|False       # Compute viscous fluxes
   vorticity = True|False            # Compute vorticity
   shock capture = True|False        # Shock capture procedure
   selective filter = True|False     # Selective filter
   selective filter n-strength = 0.7 # Strength of the filter
   selective filter 0-strength = 0.1 # Strength on the nearest point from a wall

   [figures]
   show figures = True|False          # Activate figures
   show probes = True|False           # Show probes in maps
   show bz = True|False               # Show PML in maps
   show bc = True|False               # Show bc profiles
   fps = 24                           # Framerate for movies

About *[geometry]* section
==========================

Grid
^^^^

The geometrical parameters `shape`, `origin`, `steps` must have the 
same size if provided. The parameter `bc` represents the boundary conditions of 
the computational domain. Each boundary can be set to rigid wall (`W`), 
absorbing buffer zone (`A`), or periodic condition (`P`). 
They are declared in the following order:: 

	left right front back [bottom top]

and must therefore have a length that is twice the dimension of the geometry 
(4 boundary conditions in 2d and 6 in 3d). If a periodic condition is specified 
on a side, it **must also be specified** on the opposite side. 

For instance, for a 2d geometry, one can write::

   [geometry]
   bc = AWWW
   shape = 512, 256
   origin = 80, 80
   steps = 1e-2, 2e-2

and for a 3d geometry::

   [geometry]
   bc = AWWWPP
   shape = 512, 256, 96
   origin = 80, 80, 50
   steps = 1e-2, 2e-2, 3e-2

Buffer zones
^^^^^^^^^^^^

The buffer zone parameters, i.e.::

   bz grid points = 20
   bz filter order = 3
   bz stretch order = 3
   bz stretch factor = 2

are used only when an absorbing condition (`A`) is specified. The grid is then 
stretched over `bz grid points` points by a factor `bz stretch factor` on each 
side where an absorbing condition is specified. All buffer zones in a simulation 
have the same parameters.

.. note::
   The PMLs available under **nsfds2** are replaced by buffer zones (which seem 
   to be more efficient) under **nsfds3**. PMLs could potentially reappear in 
   future **nsfds3** releases.

Obstacles and transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To customize the geometry, one can provide a set of custom obstacles to the 
:py:class:`nsfds3.cpgrid.CartesianGrid` or 
:py:class:`nsfds3.cpgrid.CurvilinearGrid` constructors. This can 
be done directly in the configuration file by providing a `geofile` containing a 
python function called `geoname` that setup a custom geometry. In the same way, 
a curvilinear transformation can be provided to the solver by setting up a 
python function `curvname`::

   [geometry]
   geofile = path_to_my_file/my_file.py
   geoname = squares
   curvname = mountain

The parameter `geofile` specifies the path to a python file that can contain 
function (or class) definitions. The solver will then search this file for 
`geoname` and/or `curvname` attributes and use them for the simulation. If these 
attributes are not found in the specified `geofile`, **nsfds3** will search for 
them in :py:class:`nsfds3.cpgrid.TestCases`. If no attributes are found, they 
will be automatically set to `None`.

The `geoname` function must take the `shape` of the grid as input argument and 
return a list of :py:class:`nsfds3.cpgrid.Obstacle` objects. For example::
	
    from nsfds3.cpgrid import Obstacle

    def squares(shape):
        """ Obstacle arangment example. """
        obs1 = Obstacle(origin=(10, 10), size=(15, 15), env=shape, bc='WWWW')
        obs2 = Obstacle(origin=(80, 70), size=(20, 20), env=shape, bc='WWWW')
        return [obs1, obs2, ]


In the same way, a curvilinear transformation can be provided to the solver by 
providing a python function `curvname` that must takes as input arguments the 
numerical axes `(xn, yn[, zn])` and returns the transformed axes 
`(xp, yp[, zp])`. For example::

    import numpy as np

    def mountain(x, y):
        """ Curvilinear function example. """
        xs = np.linspace(-np.pi, np.pi, x.shape[0])
        s = np.sin(xs / 0.1)
        profile = np.zeros_like(x)
        for i in range(x.shape[1]):
            profile[:, i] = (2 / (i / 50 + 1)) * (s - xs**2)
        return x.copy(), y + profile

About *wall sources*
====================

As mentioned previously, obstacles can be defined with 
:py:class:`nsfds3.cpgrid.Obstacle`. It is possible to specify one or more faces 
on each obstacle, which will be treated as a wall whose velocity can be set. 
To do this, the parameter `bc` of the obstacle have to be set to **V** for the 
desired side and the `set_source` method inherited by each obstacle face has to 
be called.

The `set_source` method takes as input argument a function describing the time 
evolution of the wall velocity. This function must take the time physical time 
`t` defined as::

   import numpy as np
   from nsfds3.solver import CfgSetup

   cfg = CfgSetup()
   time = np.linspace(0, cfg.nt * cfg.dt, cfg.nt + 1)

and must return a `1d numpy.array` of the same dimension as `time`.

The `set_source` method also takes the `profile` keyword argument that specifies 
the spatial profile of the boundary that can be a sine (`profile='sine'`) or a 
tappered cosine (`profile='tukey'`). For example::

    import numpy as np


    def sine(t):
        """ Sinusoïdal time evolution """
        f = 1 / (50 * t[1])
        amp = 1
        return amp * np.sin(2 * np.pi * f * t)


    def single_source(shape):
        """ Single obstacle with wall source on right face."""
        obs = Obstacle(origin=(20, 20), size=(30, 40), env=shape, bc='WVWW')
        obs.face_right.set_source(func=sine, profile='sine')
        return [obs, ]

About *[sources]* and *[initial pulses]* sections
=================================================

It is possible to declare `sources` or `initial pulse` as follows::

   [sources]
   on = True
   origins = (50, 40), (120, 80)
   amplitudes = 1e4, 2e4
   widths = 5, 4
   evolutions = 14, sine

In this example, two `sources` are specified for a 2d configuration:

	* the first one is located at `(50, 40)` with an amplitude of `1e4` Pa, a width of `5` times de spacial step, and a sinusoïdal time evolution at `14` Hz,
	* the second one is located at `(120, 80)` with an amplitude of `2e4` Pa, a width of `4` times the spacial step. and a time evolution specified by the `sine` function that must be defined in `geofile`.


Note on the use of *.wav files* as sources
==========================================

**Important:** When using a Monopole or a wall source whose time evolution is 
specified from a .wav file, you will have to resample it at the sampling 
frequency of the simulation, i.e. 1/dt. Then, pay attention to the spatial steps 
(*dx*, *dy*, *dz*) used for the simulation. 
To resolve frequencies until 20 kHz, *dx*, *dy*, *dz* must be < 0.017 m.

Examples
========

Here are some configuration examples.

- :download:`Base 2d case<examples/base_2d.conf>`
- :download:`Base 3d case <examples/base_3d.conf>`
- :download:`Curvilinear 2d case <examples/curvi_2d.conf>`
- :download:`Wall source <examples/wall_source.conf>`


To run the solver with one of these configuration::

   nsfds3 solve -c reference.conf
