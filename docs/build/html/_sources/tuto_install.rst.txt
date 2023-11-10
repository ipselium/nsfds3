Introducing **nsfds3**
======================

**nsfds3** is 3D Navier-Stokes Solver that uses Finite Difference Time Domain 
(FDTD) method. In particular, **nsfds3** is specialized in acoustic simulations.
It succeeds **nsfds2**, which only allowed 2d simulations. Note that **nsfds2** 
is now no longer maintened.

**nsfds3** is still in developpement. It is still full of bugs and comes with 
**ABSOLUTELY NO WARRANTY**.


Dependencies
------------

:python: >= 3.7
:numpy: >= 1.2
:matplotlib: >= 3.6
:scipy: >= 1.8
:h5py: >= 2.8
:rich: >= 13.1

**Important:** To create animations (using `nsfds3 make movie` for example), you
also need to have **ffmpeg** installed on your system.


Installation
------------

Clone the repo at https://github.com/ipselium/nsfds3 and::

   python setup.py install

or install it from pypi.org::

   pip install nsfds3

If pip version is >= 23.0, one can do::
   
   pip install nsfds3 --break-system-packages 
   


**Note:** To compile the dependency *libfds*, OS X users may recquire :

::

   xcode-select --install
