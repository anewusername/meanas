"""
Sparse matrix operators for use with electromagnetic wave equations.

These functions return sparse-matrix (scipy.sparse.spmatrix) representations of
 a variety of operators, intended for use with E and H fields vectorized using the
 fdfd_tools.vec() and .unvec() functions (column-major/Fortran ordering).

E- and H-field values are defined on a Yee cell; epsilon values should be calculated for
 cells centered at each E component (mu at each H component).

Many of these functions require a 'dxes' parameter, of type fdfd_tools.dx_lists_type,
 which contains grid cell width information in the following format:
 [[[dx_e_0, dx_e_1, ...], [dy_e_0, ...], [dz_e_0, ...]],
  [[dx_h_0, dx_h_1, ...], [dy_h_0, ...], [dz_h_0, ...]]]
 where dx_e_0 is the x-width of the x=0 cells, as used when calculating dE/dx,
 and dy_h_0 is  the y-width of the y=0 cells, as used when calculating dH/dy, etc.


The following operators are included:
- E-only wave operator
- H-only wave operator
- EH wave operator
- Curl for use with E, H fields
- E to H conversion
- M to J conversion
- Poynting cross products

Also available:
- Circular shifts
- Discrete derivatives
- Averaging operators
- Cross product matrices
"""

from typing import List, Tuple
import numpy
import scipy.sparse as sparse

from . import vec, dx_lists_t, vfield_t


__author__ = 'Jan Petykiewicz'


def e_full(omega: complex,
           dxes: dx_lists_t,
           epsilon: vfield_t,
           mu: vfield_t = None,
           pec: vfield_t = None,
           pmc: vfield_t = None,
           ) -> sparse.spmatrix:
    """
    Wave operator del x (1/mu * del x) - omega**2 * epsilon, for use with E-field,
     with wave equation
    (del x (1/mu * del x) - omega**2 * epsilon) E = -i * omega * J

    To make this matrix symmetric, use the preconditions from e_full_preconditioners().

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Vectorized dielectric constant
    :param mu: Vectorized magnetic permeability (default 1 everywhere).
    :param pec: Vectorized mask specifying PEC cells. Any cells where pec != 0 are interpreted
        as containing a perfect electrical conductor (PEC).
        The PEC is applied per-field-component (ie, pec.size == epsilon.size)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :return: Sparse matrix containing the wave operator
    """
    ce = curl_e(dxes)
    ch = curl_h(dxes)

    if numpy.any(numpy.equal(pec, None)):
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(numpy.where(pec, 0, 1))     # Set pe to (not PEC)

    if numpy.any(numpy.equal(pmc, None)):
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(numpy.where(pmc, 0, 1))     # set pm to (not PMC)

    e = sparse.diags(epsilon)
    if numpy.any(numpy.equal(mu, None)):
        m_div = sparse.eye(epsilon.size)
    else:
        m_div = sparse.diags(1 / mu)

    op = pe @ (ch @ pm @ m_div @ ce - omega**2 * e) @ pe
    return op


def e_full_preconditioners(dxes: dx_lists_t
                           ) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    """
    Left and right preconditioners (Pl, Pr) for symmetrizing the e_full wave operator.

    The preconditioned matrix A_symm = (Pl @ A @ Pr) is complex-symmetric
     (non-Hermitian unless there is no loss or PMLs).

    The preconditioner matrices are diagonal and complex, with Pr = 1 / Pl

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Preconditioner matrices (Pl, Pr)
    """
    p_squared = [dxes[0][0][:, None, None] * dxes[1][1][None, :, None] * dxes[1][2][None, None, :],
                 dxes[1][0][:, None, None] * dxes[0][1][None, :, None] * dxes[1][2][None, None, :],
                 dxes[1][0][:, None, None] * dxes[1][1][None, :, None] * dxes[0][2][None, None, :]]

    p_vector = numpy.sqrt(vec(p_squared))
    P_left = sparse.diags(p_vector)
    P_right = sparse.diags(1 / p_vector)
    return P_left, P_right


def h_full(omega: complex,
           dxes: dx_lists_t,
           epsilon: vfield_t,
           mu: vfield_t = None,
           pec: vfield_t = None,
           pmc: vfield_t = None,
           ) -> sparse.spmatrix:
    """
    Wave operator del x (1/epsilon * del x) - omega**2 * mu, for use with H-field,
     with wave equation
    (del x (1/epsilon * del x) - omega**2 * mu) H = i * omega * M

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Vectorized dielectric constant
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :param pec: Vectorized mask specifying PEC cells. Any cells where pec != 0 are interpreted
        as containing a perfect electrical conductor (PEC).
        The PEC is applied per-field-component (ie, pec.size == epsilon.size)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :return: Sparse matrix containing the wave operator
    """
    ec = curl_e(dxes)
    hc = curl_h(dxes)

    if numpy.any(numpy.equal(pec, None)):
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(numpy.where(pec, 0, 1))    # set pe to (not PEC)

    if numpy.any(numpy.equal(pmc, None)):
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(numpy.where(pmc, 0, 1))    # Set pe to (not PMC)

    e_div = sparse.diags(1 / epsilon)
    if mu is None:
        m = sparse.eye(epsilon.size)
    else:
        m = sparse.diags(mu)

    A = pm @ (ec @ pe @ e_div @ hc - omega**2 * m) @ pm
    return A


def eh_full(omega: complex,
            dxes: dx_lists_t,
            epsilon: vfield_t,
            mu: vfield_t = None,
            pec: vfield_t = None,
            pmc: vfield_t = None
            ) -> sparse.spmatrix:
    """
    Wave operator for [E, H] field representation. This operator implements Maxwell's
     equations without cancelling out either E or H. The operator is
     [[-i * omega * epsilon,  del x],
      [del x, i * omega * mu]]

    for use with a field vector of the form hstack(vec(E), vec(H)).

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Vectorized dielectric constant
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :param pec: Vectorized mask specifying PEC cells. Any cells where pec != 0 are interpreted
        as containing a perfect electrical conductor (PEC).
        The PEC is applied per-field-component (ie, pec.size == epsilon.size)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :return: Sparse matrix containing the wave operator
    """
    if numpy.any(numpy.equal(pec, None)):
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(numpy.where(pec, 0, 1))    # set pe to (not PEC)

    if numpy.any(numpy.equal(pmc, None)):
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(numpy.where(pmc, 0, 1))    # set pm to (not PMC)

    iwe = pe @ (1j * omega * sparse.diags(epsilon)) @ pe
    iwm = 1j * omega
    if not numpy.any(numpy.equal(mu, None)):
        iwm *= sparse.diags(mu)
    iwm = pm @ iwm @ pm

    A1 = pe @ curl_h(dxes) @ pm
    A2 = pm @ curl_e(dxes) @ pe

    A = sparse.bmat([[-iwe, A1],
                     [A2,  iwm]])
    return A


def curl_h(dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Curl operator for use with the H field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Sparse matrix for taking the discretized curl of the H-field
    """
    return cross(deriv_back(dxes[1]))


def curl_e(dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Curl operator for use with the E field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Sparse matrix for taking the discretized curl of the E-field
    """
    return cross(deriv_forward(dxes[0]))


def e2h(omega: complex,
        dxes: dx_lists_t,
        mu: vfield_t = None,
        pmc: vfield_t = None,
        ) -> sparse.spmatrix:
    """
    Utility operator for converting the E field into the H field.
    For use with e_full -- assumes that there is no magnetic current M.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :return: Sparse matrix for converting E to H
    """
    op = curl_e(dxes) / (-1j * omega)

    if not numpy.any(numpy.equal(mu, None)):
        op = sparse.diags(1 / mu) @ op

    if not numpy.any(numpy.equal(pmc, None)):
        op = sparse.diags(numpy.where(pmc, 0, 1)) @ op

    return op


def m2j(omega: complex,
        dxes: dx_lists_t,
        mu: vfield_t = None
        ) -> sparse.spmatrix:
    """
    Utility operator for converting M field into J.
    Converts a magnetic current M into an electric current J.
    For use with eg. e_full.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :return: Sparse matrix for converting E to H
    """
    op = curl_h(dxes) / (1j * omega)

    if not numpy.any(numpy.equal(mu, None)):
        op = op @ sparse.diags(1 / mu)

    return op


def rotation(axis: int, shape: List[int], shift_distance: int=1) -> sparse.spmatrix:
    """
    Utility operator for performing a circular shift along a specified axis by 1 element.

    :param axis: Axis to shift along. x=0, y=1, z=2
    :param shape: Shape of the grid being shifted
    :param shift_distance: Number of cells to shift by. May be negative. Default 1.
    :return: Sparse matrix for performing the circular shift
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))
    if axis not in range(len(shape)):
        raise Exception('Invalid direction: {}, shape is {}'.format(axis, shape))

    shifts = [abs(shift_distance) if a == axis else 0 for a in range(3)]
    shifted_diags = [(numpy.arange(n) + s) % n for n, s in zip(shape, shifts)]
    ijk = numpy.meshgrid(*shifted_diags, indexing='ij')

    n = numpy.prod(shape)
    i_ind = numpy.arange(n)
    j_ind = numpy.ravel_multi_index(ijk, shape, order='C')

    vij = (numpy.ones(n), (i_ind, j_ind.ravel(order='C')))

    d = sparse.csr_matrix(vij, shape=(n, n))

    if shift_distance < 0:
        d = d.T

    return d


def shift_with_mirror(axis: int, shape: List[int], shift_distance: int=1) -> sparse.spmatrix:
    """
    Utility operator for performing an n-element shift along a specified axis, with mirror
    boundary conditions applied to the cells beyond the receding edge.

    :param axis: Axis to shift along. x=0, y=1, z=2
    :param shape: Shape of the grid being shifted
    :param shift_distance: Number of cells to shift by. May be negative. Default 1.
    :return: Sparse matrix for performing the circular shift
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))
    if axis not in range(len(shape)):
        raise Exception('Invalid direction: {}, shape is {}'.format(axis, shape))
    if shift_distance >= shape[axis]:
        raise Exception('Shift ({}) is too large for axis {} of size {}'.format(
                        shift_distance, axis, shape[axis]))

    def mirrored_range(n, s):
        v = numpy.arange(n) + s
        v = numpy.where(v >= n, 2 * n - v - 1, v)
        v = numpy.where(v < 0, - 1 - v, v)
        return v

    shifts = [shift_distance if a == axis else 0 for a in range(3)]
    shifted_diags = [mirrored_range(n, s) for n, s in zip(shape, shifts)]
    ijk = numpy.meshgrid(*shifted_diags, indexing='ij')

    n = numpy.prod(shape)
    i_ind = numpy.arange(n)
    j_ind = ijk[0] + ijk[1] * shape[0]
    if len(shape) == 3:
        j_ind += ijk[2] * shape[0] * shape[1]

    vij = (numpy.ones(n), (i_ind, j_ind.ravel(order='C')))

    d = sparse.csr_matrix(vij, shape=(n, n))
    return d


def deriv_forward(dx_e: List[numpy.ndarray]) -> List[sparse.spmatrix]:
    """
    Utility operators for taking discretized derivatives (forward variant).

    :param dx_e: Lists of cell sizes for all axes [[dx_0, dx_1, ...], ...].
    :return: List of operators for taking forward derivatives along each axis.
    """
    shape = [s.size for s in dx_e]
    n = numpy.prod(shape)

    dx_e_expanded = numpy.meshgrid(*dx_e, indexing='ij')

    def deriv(axis):
        return rotation(axis, shape, 1) - sparse.eye(n)

    Ds = [sparse.diags(+1 / dx.ravel(order='C')) @ deriv(a)
          for a, dx in enumerate(dx_e_expanded)]

    return Ds


def deriv_back(dx_h: List[numpy.ndarray]) -> List[sparse.spmatrix]:
    """
    Utility operators for taking discretized derivatives (backward variant).

    :param dx_h: Lists of cell sizes for all axes [[dx_0, dx_1, ...], ...].
    :return: List of operators for taking forward derivatives along each axis.
    """
    shape = [s.size for s in dx_h]
    n = numpy.prod(shape)

    dx_h_expanded = numpy.meshgrid(*dx_h, indexing='ij')

    def deriv(axis):
        return rotation(axis, shape, -1) - sparse.eye(n)

    Ds = [sparse.diags(-1 / dx.ravel(order='C')) @ deriv(a)
          for a, dx in enumerate(dx_h_expanded)]

    return Ds


def cross(B: List[sparse.spmatrix]) -> sparse.spmatrix:
    """
    Cross product operator

    :param B: List [Bx, By, Bz] of sparse matrices corresponding to the x, y, z
            portions of the operator on the left side of the cross product.
    :return: Sparse matrix corresponding to (B x), where x is the cross product
    """
    n = B[0].shape[0]
    zero = sparse.csr_matrix((n, n))
    return sparse.bmat([[zero, -B[2], B[1]],
                        [B[2], zero, -B[0]],
                        [-B[1], B[0], zero]])


def vec_cross(b: vfield_t) -> sparse.spmatrix:
    """
    Vector cross product operator

    :param b: Vector on the left side of the cross product
    :return: Sparse matrix corresponding to (b x), where x is the cross product
    """
    B = [sparse.diags(c) for c in numpy.split(b, 3)]
    return cross(B)


def avgf(axis: int, shape: List[int]) -> sparse.spmatrix:
    """
    Forward average operator (x4 = (x4 + x5) / 2)

    :param axis: Axis to average along (x=0, y=1, z=2)
    :param shape: Shape of the grid to average
    :return: Sparse matrix for forward average operation
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))

    n = numpy.prod(shape)
    return 0.5 * (sparse.eye(n) + rotation(axis, shape))


def avgb(axis: int, shape: List[int]) -> sparse.spmatrix:
    """
    Backward average operator (x4 = (x4 + x3) / 2)

    :param axis: Axis to average along (x=0, y=1, z=2)
    :param shape: Shape of the grid to average
    :return: Sparse matrix for backward average operation
    """
    return avgf(axis, shape).T


def poynting_e_cross(e: vfield_t, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Operator for computing the Poynting vector, contining the (E x) portion of the Poynting vector.

    :param e: Vectorized E-field for the ExH cross product
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Sparse matrix containing (E x) portion of Poynting cross product
    """
    shape = [len(dx) for dx in dxes[0]]

    fx, fy, fz = [avgf(i, shape) for i in range(3)]
    bx, by, bz = [avgb(i, shape) for i in range(3)]

    dxag = [dx.ravel(order='C') for dx in numpy.meshgrid(*dxes[0], indexing='ij')]
    dbgx, dbgy, dbgz = [sparse.diags(dx.ravel(order='C'))
                        for dx in numpy.meshgrid(*dxes[1], indexing='ij')]

    Ex, Ey, Ez = [sparse.diags(ei * da) for ei, da in zip(numpy.split(e, 3), dxag)]

    n = numpy.prod(shape)
    zero = sparse.csr_matrix((n, n))

    P = sparse.bmat(
        [[ zero,                -fx @ Ez @ bz @ dbgy,  fx @ Ey @ by @ dbgz],
         [ fy @ Ez @ bz @ dbgx,  zero,                -fy @ Ex @ bx @ dbgz],
         [-fz @ Ey @ by @ dbgx,  fz @ Ex @ bx @ dbgy,  zero]])
    return P


def poynting_h_cross(h: vfield_t, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Operator for computing the Poynting vector, containing the (H x) portion of the Poynting vector.

    :param h: Vectorized H-field for the HxE cross product
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Sparse matrix containing (H x) portion of Poynting cross product
    """
    shape = [len(dx) for dx in dxes[0]]

    fx, fy, fz = [avgf(i, shape) for i in range(3)]
    bx, by, bz = [avgb(i, shape) for i in range(3)]

    dxbg = [dx.ravel(order='C') for dx in numpy.meshgrid(*dxes[1], indexing='ij')]
    dagx, dagy, dagz = [sparse.diags(dx.ravel(order='C'))
                        for dx in numpy.meshgrid(*dxes[0], indexing='ij')]

    Hx, Hy, Hz = [sparse.diags(hi * db) for hi, db in zip(numpy.split(h, 3), dxbg)]

    n = numpy.prod(shape)
    zero = sparse.csr_matrix((n, n))

    P = sparse.bmat(
        [[ zero,                -by @ Hz @ fx @ dagy,  bz @ Hy @ fx @ dagz],
         [ bx @ Hz @ fy @ dagx,  zero,                -bz @ Hx @ fy @ dagz],
         [-bx @ Hy @ fz @ dagx,  by @ Hx @ fz @ dagy,  zero]])
    return P
