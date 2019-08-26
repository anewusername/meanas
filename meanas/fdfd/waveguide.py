"""
Various operators and helper functions for solving for waveguide modes.

Assuming a z-dependence of the from exp(-i * wavenumber * z), we can simplify Maxwell's
 equations in the absence of sources to the form

A @ [H_x, H_y] = wavenumber**2 * [H_x, H_y]

with A =
omega**2 * epsilon * mu +
epsilon * [[-Dy], [Dx]] / epsilon * [-Dy, Dx] +
[[Dx], [Dy]] / mu * [Dx, Dy] * mu

which is the form used in this file.

As the z-dependence is known, all the functions in this file assume a 2D grid
 (ie. dxes = [[[dx_e_0, dx_e_1, ...], [dy_e_0, ...]], [[dx_h_0, ...], [dy_h_0, ...]]])
 with propagation along the z axis.
"""
# TODO update module docs

from typing import List, Tuple
import numpy
from numpy.linalg import norm
import scipy.sparse as sparse

from .. import vec, unvec, dx_lists_t, field_t, vfield_t
from . import operators


__author__ = 'Jan Petykiewicz'


def operator_e(omega: complex,
             dxes: dx_lists_t,
             epsilon: vfield_t,
             mu: vfield_t = None,
             ) -> sparse.spmatrix:
    if numpy.any(numpy.equal(mu, None)):
        mu = numpy.ones_like(epsilon)

    Dfx, Dfy = operators.deriv_forward(dxes[0])
    Dbx, Dby = operators.deriv_back(dxes[1])

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
               epsilon: vfield_t,
               mu: vfield_t = None,
               ) -> sparse.spmatrix:
    """
    Waveguide operator of the form

    omega**2 * epsilon * mu +
    epsilon * [[-Dy], [Dx]] / epsilon * [-Dy, Dx] +
    [[Dx], [Dy]] / mu * [Dx, Dy] * mu

    for use with a field vector of the form [H_x, H_y].

    This operator can be used to form an eigenvalue problem of the form
    A @ [H_x, H_y] = wavenumber**2 * [H_x, H_y]

    which can then be solved for the eigenmodes of the system (an exp(-i * wavenumber * z)
    z-dependence is assumed for the fields).

    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representation of the operator
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
                        epsilon: vfield_t,
                        mu: vfield_t = None,
                        dx_prop: float = 0,
                        ) -> Tuple[vfield_t, vfield_t]:
    """
    Given a vector e_xy containing the vectorized E_x and E_y fields,
     returns normalized, vectorized E and H fields for the system.

    :param e_xy: Vector containing E_x and E_y fields
    :param wavenumber: Wavenumber satisfying `operator_e(...) @ e_xy == wavenumber**2 * e_xy`
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :param dxes_prop: Grid cell width in the propagation direction. Default 0 (continuous).
    :return: Normalized, vectorized (e, h) containing all vector components.
    """
    e = exy2e(wavenumber=wavenumber, dxes=dxes, epsilon=epsilon) @ e_xy
    h = exy2h(wavenumber=wavenumber, omega=omega, dxes=dxes, epsilon=epsilon, mu=mu) @ e_xy
    e_norm, h_norm = _normalized_fields(e=e, h=h, wavenumber=wavenumber, omega=omega,
                                        dxes=dxes, epsilon=epsilon, mu=mu, dx_prop=dx_prop)
    return e_norm, h_norm


def normalized_fields_h(h_xy: numpy.ndarray,
                        wavenumber: complex,
                        omega: complex,
                        dxes: dx_lists_t,
                        epsilon: vfield_t,
                        mu: vfield_t = None,
                        dx_prop: float = 0,
                        ) -> Tuple[vfield_t, vfield_t]:
    """
    Given a vector e_xy containing the vectorized E_x and E_y fields,
     returns normalized, vectorized E and H fields for the system.

    :param e_xy: Vector containing E_x and E_y fields
    :param wavenumber: Wavenumber satisfying `operator_e(...) @ e_xy == wavenumber**2 * e_xy`
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :param dxes_prop: Grid cell width in the propagation direction. Default 0 (continuous).
    :return: Normalized, vectorized (e, h) containing all vector components.
    """
    e = hxy2e(wavenumber=wavenumber, omega=omega, dxes=dxes, epsilon=epsilon, mu=mu) @ h_xy
    h = hxy2h(wavenumber=wavenumber, dxes=dxes, mu=mu) @ h_xy
    e_norm, h_norm = _normalized_fields(e=e, h=h, wavenumber=wavenumber, omega=omega,
                                        dxes=dxes, epsilon=epsilon, mu=mu, dx_prop=dx_prop)
    return e_norm, h_norm


def _normalized_fields(e: numpy.ndarray,
                       h: numpy.ndarray,
                       wavenumber: complex,
                       omega: complex,
                       dxes: dx_lists_t,
                       epsilon: vfield_t,
                       mu: vfield_t = None,
                       dx_prop: float = 0,
                       ) -> Tuple[vfield_t, vfield_t]:
    # TODO documentation
    shape = [s.size for s in dxes[0]]
    dxes_real = [[numpy.real(d) for d in numpy.meshgrid(*dxes[v], indexing='ij')] for v in (0, 1)]

    E = unvec(e, shape)
    H = unvec(h, shape)

    phase = numpy.exp(-1j * wavenumber * dx_prop / 2)
    S1 = E[0] * numpy.conj(H[1] * phase) * dxes_real[0][1] * dxes_real[1][0]
    S2 = E[1] * numpy.conj(H[0] * phase) * dxes_real[0][0] * dxes_real[1][1]
    P = numpy.real(S1.sum() - S2.sum())
    assert P > 0, 'Found a mode propagating in the wrong direction! P={}'.format(P)

    energy = epsilon * e.conj() * e

    norm_amplitude = 1 / numpy.sqrt(P)
    norm_angle = -numpy.angle(e[energy.argmax()])       # Will randomly add a negative sign when mode is symmetric

    # Try to break symmetry to assign a consistent sign [experimental]
    E_weighted = unvec(e * energy * numpy.exp(1j * norm_angle), shape)
    sign = numpy.sign(E_weighted[:, :max(shape[0]//2, 1), :max(shape[1]//2, 1)].real.sum())

    norm_factor = sign * norm_amplitude * numpy.exp(1j * norm_angle)

    e *= norm_factor
    h *= norm_factor

    return e, h


def exy2h(wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfield_t,
          mu: vfield_t = None
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector e_xy containing the vectorized E_x and E_y fields,
     into a vectorized H containing all three H components

    :param wavenumber: Wavenumber satisfying `operator_e(...) @ e_xy == wavenumber**2 * e_xy`
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representing the operator
    """
    e2hop = e2h(wavenumber=wavenumber, omega=omega, dxes=dxes, mu=mu)
    return e2hop @ exy2e(wavenumber=wavenumber, dxes=dxes, epsilon=epsilon)


def hxy2e(wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfield_t,
          mu: vfield_t = None
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector h_xy containing the vectorized H_x and H_y fields,
     into a vectorized E containing all three E components

    :param wavenumber: Wavenumber satisfying `operator_h(...) @ h_xy == wavenumber**2 * h_xy`
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representing the operator
    """
    h2eop = h2e(wavenumber=wavenumber, omega=omega, dxes=dxes, epsilon=epsilon)
    return h2eop @ hxy2h(wavenumber=wavenumber, dxes=dxes, mu=mu)


def hxy2h(wavenumber: complex,
          dxes: dx_lists_t,
          mu: vfield_t = None
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector h_xy containing the vectorized H_x and H_y fields,
     into a vectorized H containing all three H components

    :param wavenumber: Wavenumber satisfying `operator_h(...) @ h_xy == wavenumber**2 * h_xy`
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representing the operator
    """
    Dfx, Dfy = operators.deriv_forward(dxes[0])
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
          epsilon: vfield_t,
          ) -> sparse.spmatrix:
    """
    Operator which transforms the vector e_xy containing the vectorized E_x and E_y fields,
     into a vectorized E containing all three E components

    :param wavenumber: Wavenumber satisfying `operator_e(...) @ e_xy == wavenumber**2 * e_xy`
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :return: Sparse matrix representing the operator
    """
    Dbx, Dby = operators.deriv_back(dxes[1])
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
        mu: vfield_t = None
        ) -> sparse.spmatrix:
    """
    Returns an operator which, when applied to a vectorized E eigenfield, produces
     the vectorized H eigenfield.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representation of the operator
    """
    op = curl_e(wavenumber, dxes) / (-1j * omega)
    if not numpy.any(numpy.equal(mu, None)):
        op = sparse.diags(1 / mu) @ op
    return op


def h2e(wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfield_t
        ) -> sparse.spmatrix:
    """
    Returns an operator which, when applied to a vectorized H eigenfield, produces
     the vectorized E eigenfield.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :return: Sparse matrix representation of the operator
    """
    op = sparse.diags(1 / (1j * omega * epsilon)) @ curl_h(wavenumber, dxes)
    return op


def curl_e(wavenumber: complex, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Discretized curl operator for use with the waveguide E field.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :return: Sparse matrix representation of the operator
    """
    n = 1
    for d in dxes[0]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dfx, Dfy = operators.deriv_forward(dxes[0])
    return operators.cross([Dfx, Dfy, Bz])


def curl_h(wavenumber: complex, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Discretized curl operator for use with the waveguide H field.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :return: Sparse matrix representation of the operator
    """
    n = 1
    for d in dxes[1]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dbx, Dby = operators.deriv_back(dxes[1])
    return operators.cross([Dbx, Dby, Bz])


def h_err(h: vfield_t,
          wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfield_t,
          mu: vfield_t = None
          ) -> float:
    """
    Calculates the relative error in the H field

    :param h: Vectorized H field
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Relative error norm(OP @ h) / norm(h)
    """
    ce = curl_e(wavenumber, dxes)
    ch = curl_h(wavenumber, dxes)

    eps_inv = sparse.diags(1 / epsilon)

    if numpy.any(numpy.equal(mu, None)):
        op = ce @ eps_inv @ ch @ h - omega ** 2 * h
    else:
        op = ce @ eps_inv @ ch @ h - omega ** 2 * (mu * h)

    return norm(op) / norm(h)


def e_err(e: vfield_t,
          wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfield_t,
          mu: vfield_t = None
          ) -> float:
    """
    Calculates the relative error in the E field

    :param e: Vectorized E field
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Relative error norm(OP @ e) / norm(e)
    """
    ce = curl_e(wavenumber, dxes)
    ch = curl_h(wavenumber, dxes)

    if numpy.any(numpy.equal(mu, None)):
        op = ch @ ce @ e - omega ** 2 * (epsilon * e)
    else:
        mu_inv = sparse.diags(1 / mu)
        op = ch @ mu_inv @ ce @ e - omega ** 2 * (epsilon * e)

    return norm(op) / norm(e)


def cylindrical_operator(omega: complex,
                         dxes: dx_lists_t,
                         epsilon: vfield_t,
                         r0: float,
                         ) -> sparse.spmatrix:
    """
    Cylindrical coordinate waveguide operator of the form

    TODO

    for use with a field vector of the form [E_r, E_y].

    This operator can be used to form an eigenvalue problem of the form
    A @ [E_r, E_y] = wavenumber**2 * [E_r, E_y]

    which can then be solved for the eigenmodes of the system (an exp(-i * wavenumber * theta)
     theta-dependence is assumed for the fields).

    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param r0: Radius of curvature for the simulation. This should be the minimum value of
        r within the simulation domain.
    :return: Sparse matrix representation of the operator
    """

    Dfx, Dfy = operators.deriv_forward(dxes[0])
    Dbx, Dby = operators.deriv_back(dxes[1])

    rx = r0 + numpy.cumsum(dxes[0][0])
    ry = r0 + dxes[0][0]/2.0 + numpy.cumsum(dxes[1][0])
    tx = rx/r0
    ty = ry/r0

    Tx = sparse.diags(vec(tx[:, None].repeat(dxes[0][1].size, axis=1)))
    Ty = sparse.diags(vec(ty[:, None].repeat(dxes[1][1].size, axis=1)))

    eps_parts = numpy.split(epsilon, 3)
    eps_x = sparse.diags(eps_parts[0])
    eps_y = sparse.diags(eps_parts[1])
    eps_z_inv = sparse.diags(1 / eps_parts[2])

    pa = sparse.vstack((Dfx, Dfy)) @ Tx @ eps_z_inv @ sparse.hstack((Dbx, Dby))
    pb = sparse.vstack((Dfx, Dfy)) @ Tx @ eps_z_inv @ sparse.hstack((Dby, Dbx))
    a0 = Ty @ eps_x + omega**-2 * Dby @ Ty @ Dfy
    a1 = Tx @ eps_y + omega**-2 * Dbx @ Ty @ Dfx
    b0 = Dbx @ Ty @ Dfy
    b1 = Dby @ Ty @ Dfx

    diag = sparse.block_diag
    op = (omega**2 * diag((Tx, Ty)) + pa) @ diag((a0, a1)) + \
        - (sparse.bmat(((None, Ty), (Tx, None))) + omega**-2 * pb) @ diag((b0, b1))

    return op



