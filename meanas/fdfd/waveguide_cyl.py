"""
Operators and helper functions for cylindrical waveguides with unchanging cross-section.

WORK IN PROGRESS, CURRENTLY BROKEN

As the z-dependence is known, all the functions in this file assume a 2D grid
 (i.e. `dxes = [[[dr_e_0, dx_e_1, ...], [dy_e_0, ...]], [[dr_h_0, ...], [dy_h_0, ...]]]`).
"""
# TODO update module docs

from typing import Dict, Union
import numpy
import scipy.sparse as sparse       # type: ignore

from ..fdmath import vec, unvec, dx_lists_t, fdfield_t, vfdfield_t, cfdfield_t
from ..fdmath.operators import deriv_forward, deriv_back
from ..eigensolvers import signed_eigensolve, rayleigh_quotient_iteration


def cylindrical_operator(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        r0: float,
        ) -> sparse.spmatrix:
    """
    Cylindrical coordinate waveguide operator of the form

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
        r0: Radius of curvature for the simulation. This should be the minimum value of
            r within the simulation domain.

    Returns:
        Sparse matrix representation of the operator
    """

    Dfx, Dfy = deriv_forward(dxes[0])
    Dbx, Dby = deriv_back(dxes[1])

    rx = r0 + numpy.cumsum(dxes[0][0])
    ry = r0 + dxes[0][0] / 2.0 + numpy.cumsum(dxes[1][0])
    tx = rx / r0
    ty = ry / r0

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

    omega2 = omega * omega

    op = (omega2 * diag((Tx, Ty)) + pa) @ diag((a0, a1)) + \
        - (sparse.bmat(((None, Ty), (Tx, None))) + pb / omega2) @ diag((b0, b1))
    return op


def solve_mode(
        mode_number: int,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        r0: float,
        ) -> Dict[str, Union[complex, cfdfield_t]]:
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
        r0: Radius of curvature for the simulation. This should be the minimum value of
            r within the simulation domain.

    Returns:
        ```
        {
            'E': List[NDArray[numpy.complex_]],
            'H': List[NDArray[numpy.complex_]],
            'wavenumber': complex,
        }
        ```
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
    '''
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]

    A_r = cylindrical_operator(numpy.real(omega), dxes_real, numpy.real(epsilon), r0)
    eigvals, eigvecs = signed_eigensolve(A_r, mode_number + 3)
    e_xy = eigvecs[:, -(mode_number + 1)]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = cylindrical_operator(omega, dxes, epsilon, r0)
    eigval, e_xy = rayleigh_quotient_iteration(A, e_xy)

    # Calculate the wave-vector (force the real part to be positive)
    wavenumber = numpy.sqrt(eigval)
    wavenumber *= numpy.sign(numpy.real(wavenumber))

    # TODO: Perform correction on wavenumber to account for numerical dispersion.

    shape = [d.size for d in dxes[0]]
    e_xy = numpy.hstack((e_xy, numpy.zeros(shape[0] * shape[1])))
    fields = {
        'wavenumber': wavenumber,
        'E': unvec(e_xy, shape),
        # 'E': unvec(e, shape),
        # 'H': unvec(h, shape),
    }

    return fields
