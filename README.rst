Introducing nsfds3
==================

|Pypi| |Licence| |DOI|


.. image:: https://github.com/ipselium/nsfds3/blob/master/docs/source/images/nsfds3.png


**nsfds3** is 3D Navier-Stokes Solver that uses Finite Difference Time Domain method.
**nsfds3** is specialized in acoustic simulations.

**nsfds3** is still in development. It is still full of bugs and comes with
**ABSOLUTELY NO WARRANTY**.


Dependencies
------------

:python: >= 3.7
:numpy: >= 1.1
:matplotlib: >= 3.0
:h5py: >= 2.8
:progressbar33: >= 2.4
:libfds: >= 0.1.0
:mplutils: >= 0.4.2

**Important:** To create animations using **nsfds3 make movie**, you also need to
have **ffmpeg** installed on your system.


Installation
------------

To install **nsfds3**:

.. code:: console

   $ pip install nsfds3 --break-system-packages


**Note:** To compile *libfds*, OS X users may require:

.. code:: console

   $ xcode-select --install


Links
-----

- **Documentation:** http://perso.univ-lemans.fr/~cdesjouy/nsfds3/
- **Source code:** https://github.com/ipselium/nsfds3
- **Bug reports:** https://github.com/ipselium/nsfds3/issues


.. |Pypi| image:: https://badge.fury.io/py/nsfds3.svg
    :target: https://pypi.org/project/nsfds3
    :alt: Pypi Package

.. |Licence| image:: https://img.shields.io/github/license/ipselium/nsfds3.svg

.. |DOI| image:: https://zenodo.org/badge/178000566.svg
    :target: https://zenodo.org/badge/latestdoi/178000566

