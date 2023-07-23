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
# Creation Date : 2023-03-16 - 22:32:44
"""
-----------

Some utils to analyze finite differences schemes.

-----------
"""

from fractions import Fraction
import numpy as np


def fd_coefficients(stencil=(-2, -1, 1, 2), order=1, verbose=False):
    '''
    Return coefficient of FD scheme for a given stencil s

    Parameters
    ----------
    stencil : sequence, optional
        Stencil of the scheme
    order : int, optional
        Order of the derivative
    verbose : boolean, optional
        Display the coefficients

    Source
    -------
    http://web.media.mit.edu/~crtaylor/calculator.html
    s_1^n . c1 + ... + s_N^n = (d!/h^d) . delta(n-d)  for 0 leq n leq N-1
    '''

    S = np.ones((len(stencil), len(stencil)))
    S[1:, :] = stencil
    for i in range(1, len(stencil)):
        S[i, :] = S[i, :]**i

    dn = np.zeros(len(stencil))
    dn[order] = np.math.factorial(order)
    coefficients = np.linalg.solve(S, dn)

    if verbose:
        sol = ''
        for c in coefficients:
            sol += str(Fraction(c).limit_denominator()) + '  '
        print(f'Stencil : {sol}')

    return dict(zip(stencil, coefficients))


def k_effective(coefficients, N=200):
    ''' Effective wavenumber k.dx

    Parameters
    ----------

    coefficients : dict
        Dictionary containing {stencil point : coefficient, ...}
    N : int, optional
        Number of point of physical wavenumber

    Returns
    -------

    kp : np.ndarray
        Physical wavenumber
    kn : np.ndarray
        Numerical wavenumber
    '''

    kp = np.linspace(1e-10, np.pi, N)
    kn = np.zeros(len(kp), dtype=complex)

    for i, k in enumerate(kp):
        for s, c in coefficients.items():
            kn[i] += -1j * c * np.exp(1j * s * k)

    return kp, kn


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    c2c = fd_coefficients([-1, 0, 1])
    c2u = fd_coefficients([0, 1, 2])

    kp, k2c = k_effective(c2c)
    kp, k2u = k_effective(c2u)

    fig, axes = plt.subplots(1, 2,  figsize=(12, 4), tight_layout=True)

    axes[0].plot(kp, kp, 'k')
    axes[0].plot(kp, k2c.real, label='FD2c')
    axes[0].plot(kp, k2u.real, label='FD2u')
    axes[0].set_title('Dispersion')
    axes[0].set_ylabel(r'$\Re(k_p\Delta x)$')

    axes[1].plot(kp, k2c.imag, label='FD2c')
    axes[1].plot(kp, k2u.imag, label='FD2u')
    axes[1].set_title('Dissipation')
    axes[1].set_ylabel(r'$\Im(k_p\Delta x)$')

    for ax in axes:
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'$k_n\Delta x$')

    plt.show()
