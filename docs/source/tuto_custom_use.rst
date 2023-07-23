==========
Custom Use
==========

Introduction
============

**nsfds3** can also be used as a classical Python package. It provides
several main objects to perform numerical simulations :

- `cpgrid` package provides objects related to the grid generation,
- `solver` package povides objects related to the solver itself,
- `graphics` module provides in particular :py:class:`nsfds3.graphics.MPLViewer` for result inspection,
- `utils` package provides various utilities.


 The following example gives the general philosophy to use **nsfds3**::

   from nsfds3.solver import CfgSetup, FDTD
   from nsfds3.cpgrid import build_mesh

   # Initialize simulation parameter
   cfg = CfgSetup()    # or cfg = CfgSetup('path_to_configfile.conf')

   # Define the mesh
   msh = build_mesh(cfg)

   # Create and run simulation
   fdtd = FDTD(msh, cfg)
   fdtd.run()

   # Figures
   fdtd.show(view='p', nans=True, buffer=True, grid=False, obstacles=True)


Once the simulation is complete, you can access the acoustic field at the last
iteration through the fdtd.fld object using the following attributes.

+---------------------+--------------------------------------------------------------------------+
| fdtd.fld attributes | description                                                              |
+=====================+==========================================================================+
| r                   | density                                                                  |
+---------------------+--------------------------------------------------------------------------+
| ru                  | product of density and x component of velocity                           |
+---------------------+--------------------------------------------------------------------------+
| rv                  | product of density and y component of velocity                           |
+---------------------+--------------------------------------------------------------------------+
| rw                  | product of density and z component of velocity (for 3d computations)     |
+---------------------+--------------------------------------------------------------------------+
| re                  | product of density and energy                                            |
+---------------------+--------------------------------------------------------------------------+
| wx                  | x component of vorticity (for 3d computations, if vorticity is computed) |
+---------------------+--------------------------------------------------------------------------+
| wy                  | y component of vorticity (for 3d computations, if vorticity is computed) |                              
+---------------------+--------------------------------------------------------------------------+
| wz                  | z component of vorticity (if vorticity is computed)                      |
+---------------------+--------------------------------------------------------------------------+
| Tk                  | Temperature (if viscous fluxes are computed)                             |
+---------------------+--------------------------------------------------------------------------+


nsfds3 files
============

As seen previously, each simulation can be configured using a .conf 
configuration file. nsfds3 creates also the additional following files: 

- a .cfg file containing the :py:class:`nsfds3.solver.CfgSetup` object used for the simulation,
- a .msh file containing the :py:class:`nsfds3.cpgrid.CartesianGrid` or :py:class:`nsfds3.cpgrid.CurvilinearGrid` object used for the simulation,
- a .hdf5 file containing, in particular, the fields computed during the simulation.

hdf5 files
----------

If `save fields` option is `True`, hdf5 are created. They contain the 
following variables.

+-------------------+---------------------------------------------------------------------------+
| var               | variable                                                                  |
+===================+===========================================================================+
| r_itX             | density                                                                   |
+-------------------+---------------------------------------------------------------------------+
| ru_itX            | product of density and x component of velocity                            |
+-------------------+---------------------------------------------------------------------------+
| rv_itX            | product of density and y component of velocity                            |
+-------------------+---------------------------------------------------------------------------+
| rw_itX            | product of density and z component of velocity (for 3d computations)      |
+-------------------+---------------------------------------------------------------------------+
| re_itX            | product of density and energy                                             |
+-------------------+---------------------------------------------------------------------------+
| wx_itX            | x component of vorticity (for 3d computations, if vorticity is computed)  |
+-------------------+---------------------------------------------------------------------------+
| wy_itX            | y component of vorticity (for 3d computations, if vorticity is computed)  |
+-------------------+---------------------------------------------------------------------------+
| wz_itX            | z component of vorticity (if vorticity is computed)                       |
+-------------------+---------------------------------------------------------------------------+
| x                 | x-grid                                                                    |
+-------------------+---------------------------------------------------------------------------+
| y                 | y-grid                                                                    |
+-------------------+---------------------------------------------------------------------------+
| z                 | z-grid                                                                    |
+-------------------+---------------------------------------------------------------------------+
| probe_locations   | coordinates of probes                                                     |
+-------------------+---------------------------------------------------------------------------+
| probe_values      | pressure at probe locations                                               |
+-------------------+---------------------------------------------------------------------------+

Also note that in curvilinear coordinates, additional variables are available :

+-------------------+--------------------------------------------------------+
| var               | variable                                               |
+===================+========================================================+
| xp                | physical x-grid                                        |
+-------------------+--------------------------------------------------------+
| yp                | physical y-grid                                        |
+-------------------+--------------------------------------------------------+
| zp                | physical z-grid (for 3d computations)                  |
+-------------------+--------------------------------------------------------+
| J                 | Jacobian matrix of transformation                      |
+-------------------+--------------------------------------------------------+

The `Y_itX` quantities provide the field `Y` at the `X`-th time step.

The pressure is not directly provided. To access acoustic pressure, one can use 
the :py:func:`nsfds3.utils.get_pressure` function as follows:: 

    from nsfds3.utils import get_pressure

	p = get_pressure(r=r, ru=ru, rv=rv, rw=rw, re=re, gamma=gamma)

`gamma` being the heat capacity ratio that can be accessed through the 
`CfgSetup` object (cfg.gamma).



cfg and msh pickle files
------------------------

`.cfg` and `.msh` files are also automatically created for each simulation. 
They contain the configuration and grid objects relative to the simulation.
One can use the :py:func:`nsfds3.utils.get_objects` function to load both 
files:: 

    from nsfds3.utils import get_objects

    cfg, msh = get_objects('path_to_cfg_and_msh_files', 'basename_of_these_files')

These objects contain in particular :

- msh.obstacles : the collection of obstacles
- cfg.dx, cfg.dy, cfg.dz, cfg.dt : spatial and time steps
- cfg.nx, cfg.ny, cfg.nz, cfg.nt, cfg.ns : Number of points (spatial and temporal)
- cfg.p0, cfg.rho0, cfg.T0, cfg.c0, cfg.gamma, cfg.prandtl, cfg.mu : Thermophysical parameters
- ... and many other parameters.
