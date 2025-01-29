"""
Operators and helper functions for cylindrical waveguides with unchanging cross-section.

WORK IN PROGRESS, CURRENTLY BROKEN

As the z-dependence is known, all the functions in this file assume a 2D grid
 (i.e. `dxes = [[[dr_e_0, dx_e_1, ...], [dy_e_0, ...]], [[dr_h_0, ...], [dy_h_0, ...]]]`).
"""
# TODO update module docs

from typing import Any, cast
from collections.abc import Sequence
import logging

import numpy
from numpy.typing import NDArray, ArrayLike
from scipy import sparse

from ..fdmath import vec, unvec, dx_lists_t, vfdfield_t, vcfdfield_t
from ..fdmath.operators import deriv_forward, deriv_back
from ..eigensolvers import signed_eigensolve, rayleigh_quotient_iteration
from . import waveguide_2d

logger = logging.getLogger(__name__)


def cylindrical_operator(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        rmin: float,
        ) -> sparse.spmatrix:
    """
    Cylindrical coordinate waveguide operator of the form

    (NOTE: See 10.1364/OL.33.001848)
    TODO: consider 10.1364/OE.20.021583

    TODO

    for use with a field vector of the form `[E_r, E_y]`.

    This operator can be used to form an eigenvalue problem of the form
        A @ [E_r, E_y] = wavenumber**2 * [E_r, E_y]

    which can then be solved for the eigenmodes of the system
    (an `exp(-i * wavenumber * theta)` theta-dependence is assumed for the fields).

    Args:
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        rmin: Radius at the left edge of the simulation domain (minimum 'x')

    Returns:
        Sparse matrix representation of the operator
    """

    Dfx, Dfy = deriv_forward(dxes[0])
    Dbx, Dby = deriv_back(dxes[1])

    Ta, Tb = dxes2T(dxes=dxes, rmin=rmin)

    eps_parts = numpy.split(epsilon, 3)
    eps_x = sparse.diags_array(eps_parts[0])
    eps_y = sparse.diags_array(eps_parts[1])
    eps_z_inv = sparse.diags_array(1 / eps_parts[2])

    omega2 = omega * omega
    diag = sparse.block_diag

    sq0 = omega2 * diag((Tb @ Tb @ eps_x,
                         Ta @ Ta @ eps_y))
    lin0 = sparse.vstack((-Tb @ Dby, Ta @ Dbx)) @ Tb @ sparse.hstack((-Dfy, Dfx))
    lin1 = sparse.vstack((Dfx, Dfy)) @ Ta @ eps_z_inv @ sparse.hstack((Dbx @ Tb @ eps_x,
                                                                       Dby @ Ta @ eps_y))
    # op = (
    #     # E
    #     omega * omega * mu_yx @ eps_xy
    #     + mu_yx @ sparse.vstack((-Dby, Dbx)) @ mu_z_inv @ sparse.hstack((-Dfy, Dfx))
    #     + sparse.vstack((Dfx, Dfy)) @ eps_z_inv @ sparse.hstack((Dbx, Dby)) @ eps_xy
    #     )

    op = sq0 + lin0 + lin1
    return op


def solve_modes(
        mode_numbers: Sequence[int],
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        rmin: float,
        mode_margin: int = 2,
        ) -> tuple[vcfdfield_t, NDArray[numpy.complex128]]:
    """
    TODO: fixup
    Given a 2d (r, y) slice of epsilon, attempts to solve for the eigenmode
     of the bent waveguide with the specified mode number.

    Args:
        mode_number: Number of the mode, 0-indexed
        omega: Angular frequency of the simulation
        dxes: Grid parameters [dx_e, dx_h] as described in meanas.fdmath.types.
              The first coordinate is assumed to be r, the second is y.
        epsilon: Dielectric constant
        rmin: Radius of curvature for the simulation. This should be the minimum value of
               r within the simulation domain.

    Returns:
        e_xys: NDArray of vfdfield_t specifying fields. First dimension is mode number.
        angular_wavenumbers: list of wavenumbers in 1/rad units.
    """

    #
    # Solve for the largest-magnitude eigenvalue of the real operator
    #
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]

    A_r = cylindrical_operator(numpy.real(omega), dxes_real, numpy.real(epsilon), rmin=rmin)
    eigvals, eigvecs = signed_eigensolve(A_r, max(mode_numbers) + mode_margin)
    keep_inds = -(numpy.array(mode_numbers) + 1)
    e_xys = eigvecs[:, keep_inds].T
    eigvals = eigvals[keep_inds]

    #
    # Now solve for the eigenvector of the full operator, using the real operator's
    #  eigenvector as an initial guess for Rayleigh quotient iteration.
    #
    A = cylindrical_operator(omega, dxes, epsilon, rmin=rmin)
    for nn in range(len(mode_numbers)):
        eigvals[nn], e_xys[nn, :] = rayleigh_quotient_iteration(A, e_xys[nn, :])

    # Calculate the wave-vector (force the real part to be positive)
    wavenumbers = numpy.sqrt(eigvals)
    wavenumbers *= numpy.sign(numpy.real(wavenumbers))

    # Wavenumbers assume the mode is at rmin, which is unlikely
    # Instead, return the wavenumber in inverse radians
    angular_wavenumbers = wavenumbers * cast(complex, rmin)

    order = angular_wavenumbers.argsort()[::-1]
    e_xys = e_xys[order]
    angular_wavenumbers = angular_wavenumbers[order]

    return e_xys, angular_wavenumbers


def solve_mode(
        mode_number: int,
        *args: Any,
        **kwargs: Any,
        ) -> tuple[vcfdfield_t, complex]:
    """
    Wrapper around `solve_modes()` that solves for a single mode.

    Args:
       mode_number: 0-indexed mode number to solve for
       *args: passed to `solve_modes()`
       **kwargs: passed to `solve_modes()`

    Returns:
        (e_xy, angular_wavenumber)
    """
    kwargs['mode_numbers'] = [mode_number]
    e_xys, angular_wavenumbers = solve_modes(*args, **kwargs)
    return e_xys[0], angular_wavenumbers[0]


def linear_wavenumbers(
        e_xys: vcfdfield_t,
        angular_wavenumbers: ArrayLike,
        epsilon: vfdfield_t,
        dxes: dx_lists_t,
        rmin: float,
        ) -> NDArray[numpy.complex128]:
    """
    Calculate linear wavenumbers (1/distance) based on angular wavenumbers (1/rad)
      and the mode's energy distribution.

    Args:
        e_xys: Vectorized mode fields with shape [num_modes, 2 * x *y)
        angular_wavenumbers: Angular wavenumbers corresponding to the fields in `e_xys`
        epsilon: Vectorized dielectric constant grid with shape (3, x, y)
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (minimum 'x')

    Returns:
        NDArray containing the calculated linear (1/distance) wavenumbers
    """
    angular_wavenumbers = numpy.asarray(angular_wavenumbers)
    mode_radii = numpy.empty_like(angular_wavenumbers, dtype=float)

    wavenumbers = numpy.empty_like(angular_wavenumbers)
    shape2d = (len(dxes[0][0]), len(dxes[0][1]))
    epsilon2d = unvec(epsilon, shape2d)[:2]
    grid_radii = rmin + numpy.cumsum(dxes[0][0])
    for ii in range(angular_wavenumbers.size):
        efield = unvec(e_xys[ii], shape2d, 2)
        energy = numpy.real((efield * efield.conj()) * epsilon2d)
        energy_vs_x = energy.sum(axis=(0, 2))
        mode_radii[ii] = (grid_radii * energy_vs_x).sum() / energy_vs_x.sum()

    logger.info(f'{mode_radii=}')
    lin_wavenumbers = angular_wavenumbers / mode_radii
    return lin_wavenumbers


