"""
Operators and helper functions for waveguides with unchanging cross-section.

The propagation direction is chosen to be along the z axis, and all fields
are given an implicit z-dependence of the form `exp(-1 * wavenumber * z)`.

As the z-dependence is known, all the functions in this file assume a 2D grid
 (i.e. `dxes = [[[dx_e_0, dx_e_1, ...], [dy_e_0, ...]], [[dx_h_0, ...], [dy_h_0, ...]]]`).
"""
# TODO update module docs

from typing import List, Tuple
import numpy
from numpy.linalg import norm
import scipy.sparse as sparse

from ..fdmath.operators import deriv_forward, deriv_back, curl_forward, curl_back, cross
from ..fdmath import vec, unvec, dx_lists_t, fdfield_t, vfdfield_t
from ..eigensolvers import signed_eigensolve, rayleigh_quotient_iteration
from . import operators


__author__ = 'Jan Petykiewicz'


def operator_e(omega: complex,
               dxes: dx_lists_t,
               epsilon: vfdfield_t,
               mu: vfdfield_t = None,
               ) -> sparse.spmatrix:
    """
    Waveguide operator of the form

        omega**2 * mu * epsilon +
        mu * [[-Dy], [Dx]] / mu * [-Dy, Dx] +
        [[Dx], [Dy]] / epsilon * [Dx, Dy] * epsilon

    for use with a field vector of the form `cat([E_x, E_y])`.

    More precisely, the operator is
    $$ \\omega^2 \\mu_{yx} \\epsilon_{xy} +
       \\mu_{yx} \\begin{bmatrix} -D_{by} \\\\
                                   D_{bx} \\end{bmatrix} \\mu_z^{-1}
                 \\begin{bmatrix} -D_{fy} & D_{fx} \\end{bmatrix} +
      \\begin{bmatrix} D_{fx} \\\\
                       D_{fy} \\end{bmatrix} \\epsilon_z^{-1}
                 \\begin{bmatrix} D_{bx} & D_{by} \\end{bmatrix} \\epsilon_{xy} $$

    where
    \\( \\epsilon_{xy} = \\begin{bmatrix}
                                 \\epsilon_x & 0 \\\\
                                           0 & \\epsilon_y
                                 \\end{bmatrix} \\),
    \\( \\mu_{yx} = \\begin{bmatrix}
                           \\mu_y & 0 \\\\
                                0 & \\mu_x
                           \\end{bmatrix} \\),
    \\( D_{fx} \\) and \\( D_{bx} \\) are the forward and backward derivatives along x,
    and each \\( \\epsilon_x, \\mu_y, \\) etc. is a diagonal matrix representing


    This operator can be used to form an eigenvalue problem of the form
    `operator_e(...) @ [E_x, E_y] = wavenumber**2 * [E_x, E_y]`

    which can then be solved for the eigenmodes of the system (an `exp(-i * wavenumber * z)`
    z-dependence is assumed for the fields).

    Args:
        omega: The angular frequency of the system.
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representation of the operator.
    """
    if numpy.any(numpy.equal(mu, None)):
        mu = numpy.ones_like(epsilon)

    Dfx, Dfy = deriv_forward(dxes[0])
    Dbx, Dby = deriv_back(dxes[1])

    eps_parts = numpy.split(epsilon, 3)
    eps_xy = sparse.diags(numpy.hstack((eps_parts[0], eps_parts[1])))
    eps_z_inv = sparse.diags(1 / eps_parts[2])

    mu_parts = numpy.split(mu, 3)
    mu_yx = sparse.diags(numpy.hstack((mu_parts[1], mu_parts[0])))
    mu_z_inv = sparse.diags(1 / mu_parts[2])

    op = omega * omega * mu_yx @ eps_xy + \
        mu_yx @ sparse.vstack((-Dby, Dbx)) @ mu_z_inv @ sparse.hstack((-Dfy, Dfx)) + \
        sparse.vstack((Dfx, Dfy)) @ eps_z_inv @ sparse.hstack((Dbx, Dby)) @ eps_xy
    return op


def operator_h(omega: complex,
               dxes: dx_lists_t,
               epsilon: vfdfield_t,
               mu: vfdfield_t = None,
               ) -> sparse.spmatrix:
    """
    Waveguide operator of the form

        omega**2 * epsilon * mu +
        epsilon * [[-Dy], [Dx]] / epsilon * [-Dy, Dx] +
        [[Dx], [Dy]] / mu * [Dx, Dy] * mu

    for use with a field vector of the form `cat([H_x, H_y])`.

    More precisely, the operator is
    $$ \\omega^2 \\epsilon_{yx} \\mu_{xy} +
       \\epsilon_{yx} \\begin{bmatrix} -D_{fy} \\\\
                                        D_{fx} \\end{bmatrix} \\epsilon_z^{-1}
                 \\begin{bmatrix} -D_{by} & D_{bx} \\end{bmatrix} +
      \\begin{bmatrix} D_{bx} \\\\
                       D_{by} \\end{bmatrix} \\mu_z^{-1}
                 \\begin{bmatrix} D_{fx} & D_{fy} \\end{bmatrix} \\mu_{xy} $$

    where
    \\( \\epsilon_{yx} = \\begin{bmatrix}
                                 \\epsilon_y & 0 \\\\
                                           0 & \\epsilon_x
                                 \\end{bmatrix} \\),
    \\( \\mu_{xy} = \\begin{bmatrix}
                           \\mu_x & 0 \\\\
                                0 & \\mu_y
                           \\end{bmatrix} \\),
    \\( D_{fx} \\) and \\( D_{bx} \\) are the forward and backward derivatives along x,
    and each \\( \\epsilon_x, \\mu_y, \\) etc. is a diagonal matrix.


    This operator can be used to form an eigenvalue problem of the form
    `operator_h(...) @ [H_x, H_y] = wavenumber**2 * [H_x, H_y]`

    which can then be solved for the eigenmodes of the system (an `exp(-i * wavenumber * z)`
    z-dependence is assumed for the fields).

    Args:
        omega: The angular frequency of the system.
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representation of the operator.
    """
    if numpy.any(numpy.equal(mu, None)):
        mu = numpy.ones_like(epsilon)

    Dfx, Dfy = operators.deriv_forward(dxes[0])
    Dbx, Dby = operators.deriv_back(dxes[1])

    eps_parts = numpy.split(epsilon, 3)
    eps_yx = sparse.diags(numpy.hstack((eps_parts[1], eps_parts[0])))
    eps_z_inv = sparse.diags(1 / eps_parts[2])

    mu_parts = numpy.split(mu, 3)
    mu_xy = sparse.diags(numpy.hstack((mu_parts[0], mu_parts[1])))
    mu_z_inv = sparse.diags(1 / mu_parts[2])

    op = omega * omega * eps_yx @ mu_xy + \
        eps_yx @ sparse.vstack((-Dfy, Dfx)) @ eps_z_inv @ sparse.hstack((-Dby, Dbx)) + \
        sparse.vstack((Dbx, Dby)) @ mu_z_inv @ sparse.hstack((Dfx, Dfy)) @ mu_xy

    return op


def normalized_fields_e(e_xy: numpy.ndarray,
                        wavenumber: complex,
                        omega: complex,
                        dxes: dx_lists_t,
                        epsilon: vfdfield_t,
                        mu: vfdfield_t = None,
                        prop_phase: float = 0,
                        ) -> Tuple[vfdfield_t, vfdfield_t]:
    """
    Given a vector `e_xy` containing the vectorized E_x and E_y fields,
     returns normalized, vectorized E and H fields for the system.

    Args:
        e_xy: Vector containing E_x and E_y fields
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`.
                    It should satisfy `operator_e() @ e_xy == wavenumber**2 * e_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)
        prop_phase: Phase shift `(dz * corrected_wavenumber)` over 1 cell in propagation direction.
                    Default 0 (continuous propagation direction, i.e. dz->0).

    Returns:
        `(e, h)`, where each field is vectorized, normalized,
        and contains all three vector components.
    """
    e = exy2e(wavenumber=wavenumber, dxes=dxes, epsilon=epsilon) @ e_xy
    h = exy2h(wavenumber=wavenumber, omega=omega, dxes=dxes, epsilon=epsilon, mu=mu) @ e_xy
    e_norm, h_norm = _normalized_fields(e=e, h=h, omega=omega, dxes=dxes, epsilon=epsilon,
                                        mu=mu, prop_phase=prop_phase)
    return e_norm, h_norm


def normalized_fields_h(h_xy: numpy.ndarray,
                        wavenumber: complex,
                        omega: complex,
                        dxes: dx_lists_t,
                        epsilon: vfdfield_t,
                        mu: vfdfield_t = None,
                        prop_phase: float = 0,
                        ) -> Tuple[vfdfield_t, vfdfield_t]:
    """
    Given a vector `h_xy` containing the vectorized H_x and H_y fields,
     returns normalized, vectorized E and H fields for the system.

    Args:
        h_xy: Vector containing H_x and H_y fields
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`.
                    It should satisfy `operator_h() @ h_xy == wavenumber**2 * h_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)
        prop_phase: Phase shift `(dz * corrected_wavenumber)` over 1 cell in propagation direction.
                    Default 0 (continuous propagation direction, i.e. dz->0).

    Returns:
        `(e, h)`, where each field is vectorized, normalized,
        and contains all three vector components.
    """
    e = hxy2e(wavenumber=wavenumber, omega=omega, dxes=dxes, epsilon=epsilon, mu=mu) @ h_xy
    h = hxy2h(wavenumber=wavenumber, dxes=dxes, mu=mu) @ h_xy
    e_norm, h_norm = _normalized_fields(e=e, h=h, omega=omega, dxes=dxes, epsilon=epsilon,
                                        mu=mu, prop_phase=prop_phase)
    return e_norm, h_norm


def _normalized_fields(e: numpy.ndarray,
                       h: numpy.ndarray,
                       omega: complex,
                       dxes: dx_lists_t,
                       epsilon: vfdfield_t,
                       mu: vfdfield_t = None,
                       prop_phase: float = 0,
                       ) -> Tuple[vfdfield_t, vfdfield_t]:
    # TODO documentation
    shape = [s.size for s in dxes[0]]
    dxes_real = [[numpy.real(d) for d in numpy.meshgrid(*dxes[v], indexing='ij')] for v in (0, 1)]

    E = unvec(e, shape)
    H = unvec(h, shape)

    # Find time-averaged Sz and normalize to it
    # H phase is adjusted by a half-cell forward shift for Yee cell, and 1-cell reverse shift for Poynting
    phase = numpy.exp(-1j * -prop_phase / 2)
    Sz_a = E[0] * numpy.conj(H[1] * phase) * dxes_real[0][1] * dxes_real[1][0]
    Sz_b = E[1] * numpy.conj(H[0] * phase) * dxes_real[0][0] * dxes_real[1][1]
    Sz_tavg = numpy.real(Sz_a.sum() - Sz_b.sum()) * 0.5       # 0.5 since E, H are assumed to be peak (not RMS) amplitudes
    assert Sz_tavg > 0, 'Found a mode propagating in the wrong direction! Sz_tavg={}'.format(Sz_tavg)

    energy = epsilon * e.conj() * e

    norm_amplitude = 1 / numpy.sqrt(Sz_tavg)
    norm_angle = -numpy.angle(e[energy.argmax()])       # Will randomly add a negative sign when mode is symmetric

    # Try to break symmetry to assign a consistent sign [experimental TODO]
    E_weighted = unvec(e * energy * numpy.exp(1j * norm_angle), shape)
    sign = numpy.sign(E_weighted[:, :max(shape[0]//2, 1), :max(shape[1]//2, 1)].real.sum())

    norm_factor = sign * norm_amplitude * numpy.exp(1j * norm_angle)

    e *= norm_factor
    h *= norm_factor

    return e, h


def exy2h(wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfdfield_t,
          mu: vfdfield_t = None
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector `e_xy` containing the vectorized E_x and E_y fields,
     into a vectorized H containing all three H components

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`.
                    It should satisfy `operator_e() @ e_xy == wavenumber**2 * e_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representing the operator.
    """
    e2hop = e2h(wavenumber=wavenumber, omega=omega, dxes=dxes, mu=mu)
    return e2hop @ exy2e(wavenumber=wavenumber, dxes=dxes, epsilon=epsilon)


def hxy2e(wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfdfield_t,
          mu: vfdfield_t = None
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector `h_xy` containing the vectorized H_x and H_y fields,
     into a vectorized E containing all three E components

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`.
                    It should satisfy `operator_h() @ h_xy == wavenumber**2 * h_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representing the operator.
    """
    h2eop = h2e(wavenumber=wavenumber, omega=omega, dxes=dxes, epsilon=epsilon)
    return h2eop @ hxy2h(wavenumber=wavenumber, dxes=dxes, mu=mu)


def hxy2h(wavenumber: complex,
          dxes: dx_lists_t,
          mu: vfdfield_t = None
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector `h_xy` containing the vectorized H_x and H_y fields,
     into a vectorized H containing all three H components

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`.
                    It should satisfy `operator_h() @ h_xy == wavenumber**2 * h_xy`
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representing the operator.
    """
    Dfx, Dfy = deriv_forward(dxes[0])
    hxy2hz = sparse.hstack((Dfx, Dfy)) / (1j * wavenumber)

    if not numpy.any(numpy.equal(mu, None)):
        mu_parts = numpy.split(mu, 3)
        mu_xy = sparse.diags(numpy.hstack((mu_parts[0], mu_parts[1])))
        mu_z_inv = sparse.diags(1 / mu_parts[2])

        hxy2hz = mu_z_inv @ hxy2hz @ mu_xy

    n_pts = dxes[1][0].size * dxes[1][1].size
    op = sparse.vstack((sparse.eye(2 * n_pts),
                        hxy2hz))
    return op


def exy2e(wavenumber: complex,
          dxes: dx_lists_t,
          epsilon: vfdfield_t,
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector `e_xy` containing the vectorized E_x and E_y fields,
     into a vectorized E containing all three E components

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
                    It should satisfy `operator_e() @ e_xy == wavenumber**2 * e_xy`
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid

    Returns:
        Sparse matrix representing the operator.
    """
    Dbx, Dby = deriv_back(dxes[1])
    exy2ez = sparse.hstack((Dbx, Dby)) / (1j * wavenumber)

    if not numpy.any(numpy.equal(epsilon, None)):
        epsilon_parts = numpy.split(epsilon, 3)
        epsilon_xy = sparse.diags(numpy.hstack((epsilon_parts[0], epsilon_parts[1])))
        epsilon_z_inv = sparse.diags(1 / epsilon_parts[2])

        exy2ez = epsilon_z_inv @ exy2ez @ epsilon_xy

    n_pts = dxes[0][0].size * dxes[0][1].size
    op = sparse.vstack((sparse.eye(2 * n_pts),
                        exy2ez))
    return op


def e2h(wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        mu: vfdfield_t = None
        ) -> sparse.spmatrix:
    """
    Returns an operator which, when applied to a vectorized E eigenfield, produces
     the vectorized H eigenfield.

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representation of the operator.
    """
    op = curl_e(wavenumber, dxes) / (-1j * omega)
    if not numpy.any(numpy.equal(mu, None)):
        op = sparse.diags(1 / mu) @ op
    return op


def h2e(wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t
        ) -> sparse.spmatrix:
    """
    Returns an operator which, when applied to a vectorized H eigenfield, produces
     the vectorized E eigenfield.

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid

    Returns:
        Sparse matrix representation of the operator.
    """
    op = sparse.diags(1 / (1j * omega * epsilon)) @ curl_h(wavenumber, dxes)
    return op


def curl_e(wavenumber: complex, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Discretized curl operator for use with the waveguide E field.

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)

    Return:
        Sparse matrix representation of the operator.
    """
    n = 1
    for d in dxes[0]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dfx, Dfy = deriv_forward(dxes[0])
    return cross([Dfx, Dfy, Bz])


def curl_h(wavenumber: complex, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Discretized curl operator for use with the waveguide H field.

    Args:
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)

    Return:
        Sparse matrix representation of the operator.
    """
    n = 1
    for d in dxes[1]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dbx, Dby = deriv_back(dxes[1])
    return cross([Dbx, Dby, Bz])


def h_err(h: vfdfield_t,
          wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfdfield_t,
          mu: vfdfield_t = None
          ) -> float:
    """
    Calculates the relative error in the H field

    Args:
        h: Vectorized H field
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Relative error `norm(A_h @ h) / norm(h)`.
    """
    ce = curl_e(wavenumber, dxes)
    ch = curl_h(wavenumber, dxes)

    eps_inv = sparse.diags(1 / epsilon)

    if numpy.any(numpy.equal(mu, None)):
        op = ce @ eps_inv @ ch @ h - omega ** 2 * h
    else:
        op = ce @ eps_inv @ ch @ h - omega ** 2 * (mu * h)

    return norm(op) / norm(h)


def e_err(e: vfdfield_t,
          wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfdfield_t,
          mu: vfdfield_t = None
          ) -> float:
    """
    Calculates the relative error in the E field

    Args:
        e: Vectorized E field
        wavenumber: Wavenumber assuming fields have z-dependence of `exp(-i * wavenumber * z)`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Relative error `norm(A_e @ e) / norm(e)`.
    """
    ce = curl_e(wavenumber, dxes)
    ch = curl_h(wavenumber, dxes)

    if numpy.any(numpy.equal(mu, None)):
        op = ch @ ce @ e - omega ** 2 * (epsilon * e)
    else:
        mu_inv = sparse.diags(1 / mu)
        op = ch @ mu_inv @ ce @ e - omega ** 2 * (epsilon * e)

    return norm(op) / norm(e)


def solve_modes(mode_numbers: List[int],
                omega: complex,
                dxes: dx_lists_t,
                epsilon: vfdfield_t,
                mu: vfdfield_t = None,
                mode_margin: int = 2,
                ) -> Tuple[List[vfdfield_t], List[complex]]:
    """
    Given a 2D region, attempts to solve for the eigenmode with the specified mode numbers.

    Args:
       mode_numbers: List of 0-indexed mode numbers to solve for
       omega: Angular frequency of the simulation
       dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
       epsilon: Dielectric constant
       mu: Magnetic permeability (default 1 everywhere)
       mode_margin: The eigensolver will actually solve for `(max(mode_number) + mode_margin)`
            modes, but only return the target mode. Increasing this value can improve the solver's
            ability to find the correct mode. Default 2.

    Returns:
        (e_xys, wavenumbers)
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
    '''
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]
    A_r = operator_e(numpy.real(omega), dxes_real, numpy.real(epsilon), numpy.real(mu))

    eigvals, eigvecs = signed_eigensolve(A_r, max(mode_numbers) + mode_margin)
    e_xys = eigvecs[:, -(numpy.array(mode_numbers) + 1)]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = operator_e(omega, dxes, epsilon, mu)
    for nn in range(len(mode_numbers)):
        eigvals[nn], e_xys[:, nn] = rayleigh_quotient_iteration(A, e_xys[:, nn])

    # Calculate the wave-vector (force the real part to be positive)
    wavenumbers = numpy.sqrt(eigvals)
    wavenumbers *= numpy.sign(numpy.real(wavenumbers))

    return e_xys, wavenumbers


def solve_mode(mode_number: int,
               *args,
               **kwargs
               ) -> Tuple[vfdfield_t, complex]:
    """
    Wrapper around `solve_modes()` that solves for a single mode.

    Args:
       mode_number: 0-indexed mode number to solve for
       *args: passed to `solve_modes()`
       **kwargs: passed to `solve_modes()`

    Returns:
        (e_xy, wavenumber)
    """
    e_xys, wavenumbers = solve_modes(mode_numbers=[mode_number], *args, **kwargs)
    return e_xys[:, 0], wavenumbers[0]
