"""
Sparse matrix operators for use with electromagnetic wave equations.

These functions return sparse-matrix (`scipy.sparse.spmatrix`) representations of
 a variety of operators, intended for use with E and H fields vectorized using the
 `meanas.fdmath.vectorization.vec()` and `meanas.fdmath.vectorization.unvec()` functions.

E- and H-field values are defined on a Yee cell; `epsilon` values should be calculated for
 cells centered at each E component (`mu` at each H component).

Many of these functions require a `dxes` parameter, of type `dx_lists_t`; see
the `meanas.fdmath.types` submodule for details.


The following operators are included:

- E-only wave operator
- H-only wave operator
- EH wave operator
- Curl for use with E, H fields
- E to H conversion
- M to J conversion
- Poynting cross products
- Circular shifts
- Discrete derivatives
- Averaging operators
- Cross product matrices
"""

from typing import Tuple, Optional
import numpy
import scipy.sparse as sparse       # type: ignore

from ..fdmath import vec, dx_lists_t, vfdfield_t, vcfdfield_t
from ..fdmath.operators import shift_with_mirror, shift_circ, curl_forward, curl_back


__author__ = 'Jan Petykiewicz'


def e_full(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        pec: Optional[vfdfield_t] = None,
        pmc: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Wave operator
     $$ \\nabla \\times (\\frac{1}{\\mu} \\nabla \\times) - \\Omega^2 \\epsilon $$

        del x (1/mu * del x) - omega**2 * epsilon

     for use with the E-field, with wave equation
     $$ (\\nabla \\times (\\frac{1}{\\mu} \\nabla \\times) - \\Omega^2 \\epsilon) E = -\\imath \\omega J $$

        (del x (1/mu * del x) - omega**2 * epsilon) E = -i * omega * J

    To make this matrix symmetric, use the preconditioners from `e_full_preconditioners()`.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Vectorized dielectric constant
        mu: Vectorized magnetic permeability (default 1 everywhere).
        pec: Vectorized mask specifying PEC cells. Any cells where `pec != 0` are interpreted
          as containing a perfect electrical conductor (PEC).
          The PEC is applied per-field-component (i.e. `pec.size == epsilon.size`)
        pmc: Vectorized mask specifying PMC cells. Any cells where `pmc != 0` are interpreted
          as containing a perfect magnetic conductor (PMC).
          The PMC is applied per-field-component (i.e. `pmc.size == epsilon.size`)

    Returns:
        Sparse matrix containing the wave operator.
    """
    ch = curl_back(dxes[1])
    ce = curl_forward(dxes[0])

    if pec is None:
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(numpy.where(pec, 0, 1))     # Set pe to (not PEC)

    if pmc is None:
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(numpy.where(pmc, 0, 1))     # set pm to (not PMC)

    e = sparse.diags(epsilon)
    if mu is None:
        m_div = sparse.eye(epsilon.size)
    else:
        m_div = sparse.diags(1 / mu)

    op = pe @ (ch @ pm @ m_div @ ce - omega**2 * e) @ pe
    return op


def e_full_preconditioners(
        dxes: dx_lists_t,
        ) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    """
    Left and right preconditioners `(Pl, Pr)` for symmetrizing the `e_full` wave operator.

    The preconditioned matrix `A_symm = (Pl @ A @ Pr)` is complex-symmetric
     (non-Hermitian unless there is no loss or PMLs).

    The preconditioner matrices are diagonal and complex, with `Pr = 1 / Pl`

    Args:
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`

    Returns:
        Preconditioner matrices `(Pl, Pr)`.
    """
    p_squared = [dxes[0][0][:, None, None] * dxes[1][1][None, :, None] * dxes[1][2][None, None, :],
                 dxes[1][0][:, None, None] * dxes[0][1][None, :, None] * dxes[1][2][None, None, :],
                 dxes[1][0][:, None, None] * dxes[1][1][None, :, None] * dxes[0][2][None, None, :]]

    p_vector = numpy.sqrt(vec(p_squared))
    P_left = sparse.diags(p_vector)
    P_right = sparse.diags(1 / p_vector)
    return P_left, P_right


def h_full(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        pec: Optional[vfdfield_t] = None,
        pmc: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Wave operator
     $$ \\nabla \\times (\\frac{1}{\\epsilon} \\nabla \\times) - \\omega^2 \\mu $$

        del x (1/epsilon * del x) - omega**2 * mu

     for use with the H-field, with wave equation
     $$ (\\nabla \\times (\\frac{1}{\\epsilon} \\nabla \\times) - \\omega^2 \\mu) E = \\imath \\omega M $$

        (del x (1/epsilon * del x) - omega**2 * mu) E = i * omega * M

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Vectorized dielectric constant
        mu: Vectorized magnetic permeability (default 1 everywhere)
        pec: Vectorized mask specifying PEC cells. Any cells where `pec != 0` are interpreted
           as containing a perfect electrical conductor (PEC).
           The PEC is applied per-field-component (i.e. `pec.size == epsilon.size`)
        pmc: Vectorized mask specifying PMC cells. Any cells where `pmc != 0` are interpreted
           as containing a perfect magnetic conductor (PMC).
           The PMC is applied per-field-component (i.e. `pmc.size == epsilon.size`)

    Returns:
        Sparse matrix containing the wave operator.
    """
    ch = curl_back(dxes[1])
    ce = curl_forward(dxes[0])

    if pec is None:
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(numpy.where(pec, 0, 1))    # set pe to (not PEC)

    if pmc is None:
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(numpy.where(pmc, 0, 1))    # Set pe to (not PMC)

    e_div = sparse.diags(1 / epsilon)
    if mu is None:
        m = sparse.eye(epsilon.size)
    else:
        m = sparse.diags(mu)

    A = pm @ (ce @ pe @ e_div @ ch - omega**2 * m) @ pm
    return A


def eh_full(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        pec: Optional[vfdfield_t] = None,
        pmc: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Wave operator for `[E, H]` field representation. This operator implements Maxwell's
     equations without cancelling out either E or H. The operator is
    $$  \\begin{bmatrix}
        -\\imath \\omega \\epsilon  &  \\nabla \\times      \\\\
        \\nabla \\times             &  \\imath \\omega \\mu
        \\end{bmatrix} $$

        [[-i * omega * epsilon,  del x         ],
         [del x,                 i * omega * mu]]

    for use with a field vector of the form `cat(vec(E), vec(H))`:
    $$  \\begin{bmatrix}
        -\\imath \\omega \\epsilon  &  \\nabla \\times      \\\\
        \\nabla \\times             &  \\imath \\omega \\mu
        \\end{bmatrix}
        \\begin{bmatrix} E \\\\
                         H
        \\end{bmatrix}
        = \\begin{bmatrix} J \\\\
                          -M
          \\end{bmatrix} $$

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Vectorized dielectric constant
        mu: Vectorized magnetic permeability (default 1 everywhere)
        pec: Vectorized mask specifying PEC cells. Any cells where `pec != 0` are interpreted
          as containing a perfect electrical conductor (PEC).
          The PEC is applied per-field-component (i.e. `pec.size == epsilon.size`)
        pmc: Vectorized mask specifying PMC cells. Any cells where `pmc != 0` are interpreted
          as containing a perfect magnetic conductor (PMC).
          The PMC is applied per-field-component (i.e. `pmc.size == epsilon.size`)

    Returns:
        Sparse matrix containing the wave operator.
    """
    if pec is None:
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(numpy.where(pec, 0, 1))    # set pe to (not PEC)

    if pmc is None:
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(numpy.where(pmc, 0, 1))    # set pm to (not PMC)

    iwe = pe @ (1j * omega * sparse.diags(epsilon)) @ pe
    iwm = 1j * omega
    if mu is not None:
        iwm *= sparse.diags(mu)
    iwm = pm @ iwm @ pm

    A1 = pe @ curl_back(dxes[1]) @ pm
    A2 = pm @ curl_forward(dxes[0]) @ pe

    A = sparse.bmat([[-iwe, A1],
                     [A2,  iwm]])
    return A


def e2h(
        omega: complex,
        dxes: dx_lists_t,
        mu: Optional[vfdfield_t] = None,
        pmc: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Utility operator for converting the E field into the H field.
    For use with `e_full()` -- assumes that there is no magnetic current M.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        mu: Vectorized magnetic permeability (default 1 everywhere)
        pmc: Vectorized mask specifying PMC cells. Any cells where `pmc != 0` are interpreted
          as containing a perfect magnetic conductor (PMC).
          The PMC is applied per-field-component (i.e. `pmc.size == epsilon.size`)

    Returns:
        Sparse matrix for converting E to H.
    """
    op = curl_forward(dxes[0]) / (-1j * omega)

    if mu is not None:
        op = sparse.diags(1 / mu) @ op

    if pmc is not None:
        op = sparse.diags(numpy.where(pmc, 0, 1)) @ op

    return op


def m2j(
        omega: complex,
        dxes: dx_lists_t,
        mu: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Operator for converting a magnetic current M into an electric current J.
    For use with eg. `e_full()`.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        mu: Vectorized magnetic permeability (default 1 everywhere)

    Returns:
        Sparse matrix for converting M to J.
    """
    op = curl_back(dxes[1]) / (1j * omega)

    if mu is not None:
        op = op @ sparse.diags(1 / mu)

    return op


def poynting_e_cross(e: vcfdfield_t, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Operator for computing the Poynting vector, containing the
    (E x) portion of the Poynting vector.

    Args:
        e: Vectorized E-field for the ExH cross product
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`

    Returns:
        Sparse matrix containing (E x) portion of Poynting cross product.
    """
    shape = [len(dx) for dx in dxes[0]]

    fx, fy, fz = [shift_circ(i, shape, 1) for i in range(3)]

    dxag = [dx.ravel(order='C') for dx in numpy.meshgrid(*dxes[0], indexing='ij')]
    Ex, Ey, Ez = [ei * da for ei, da in zip(numpy.split(e, 3), dxag)]

    block_diags = [[ None,     fx @ -Ez, fx @  Ey],
                   [ fy @  Ez, None,     fy @ -Ex],
                   [ fz @ -Ey, fz @  Ex, None]]
    block_matrix = sparse.bmat([[sparse.diags(x) if x is not None else None for x in row]
                                for row in block_diags])
    P = block_matrix @ sparse.diags(numpy.concatenate(dxag))
    return P


def poynting_h_cross(h: vcfdfield_t, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Operator for computing the Poynting vector, containing the (H x) portion of the Poynting vector.

    Args:
        h: Vectorized H-field for the HxE cross product
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`

    Returns:
        Sparse matrix containing (H x) portion of Poynting cross product.
    """
    shape = [len(dx) for dx in dxes[0]]

    fx, fy, fz = [shift_circ(i, shape, 1) for i in range(3)]

    dxag = [dx.ravel(order='C') for dx in numpy.meshgrid(*dxes[0], indexing='ij')]
    dxbg = [dx.ravel(order='C') for dx in numpy.meshgrid(*dxes[1], indexing='ij')]
    Hx, Hy, Hz = [sparse.diags(hi * db) for hi, db in zip(numpy.split(h, 3), dxbg)]

    P = (sparse.bmat(
        [[ None,    -Hz @ fx,   Hy @ fx],
         [ Hz @ fy,  None,     -Hx @ fy],
         [-Hy @ fz,  Hx @ fz,   None]])
         @ sparse.diags(numpy.concatenate(dxag)))
    return P


def e_tfsf_source(
        TF_region: vfdfield_t,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Operator that turns a desired E-field distribution into a
     total-field/scattered-field (TFSF) source.

    TODO: Reference Rumpf paper

    Args:
        TF_region: Mask, which is set to 1 inside the total-field region and 0 in the
                   scattered-field region
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Vectorized dielectric constant
        mu: Vectorized magnetic permeability (default 1 everywhere).

    Returns:
        Sparse matrix that turns an E-field into a current (J) distribution.

    """
    # TODO documentation
    A = e_full(omega, dxes, epsilon, mu)
    Q = sparse.diags(TF_region)
    return (A @ Q - Q @ A) / (-1j * omega)


def e_boundary_source(
        mask: vfdfield_t,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        periodic_mask_edges: bool = False,
        ) -> sparse.spmatrix:
    """
    Operator that turns an E-field distrubtion into a current (J) distribution
      along the edges (external and internal) of the provided mask. This is just an
      `e_tfsf_source()` with an additional masking step.

    Args:
        mask: The current distribution is generated at the edges of the mask,
              i.e. any points where shifting the mask by one cell in any direction
              would change its value.
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Vectorized dielectric constant
        mu: Vectorized magnetic permeability (default 1 everywhere).

    Returns:
        Sparse matrix that turns an E-field into a current (J) distribution.
    """
    full = e_tfsf_source(TF_region=mask, omega=omega, dxes=dxes, epsilon=epsilon, mu=mu)

    shape = [len(dxe) for dxe in dxes[0]]
    jmask = numpy.zeros_like(mask, dtype=bool)

    def shift_rot(axis: int, polarity: int) -> sparse.spmatrix:
        return shift_circ(axis=axis, shape=shape, shift_distance=polarity)

    def shift_mir(axis: int, polarity: int) -> sparse.spmatrix:
        return shift_with_mirror(axis=axis, shape=shape, shift_distance=polarity)

    shift = shift_rot if periodic_mask_edges else shift_mir

    for axis in (0, 1, 2):
        if shape[axis] == 1:
            continue
        for polarity in (-1, +1):
            r = shift(axis, polarity) - sparse.eye(numpy.prod(shape))  # shifted minus original
            r3 = sparse.block_diag((r, r, r))
            jmask = numpy.logical_or(jmask, numpy.abs(r3 @ mask))

#    jmask = ((numpy.roll(mask, -1, axis=0) != mask) |
#             (numpy.roll(mask, +1, axis=0) != mask) |
#             (numpy.roll(mask, -1, axis=1) != mask) |
#             (numpy.roll(mask, +1, axis=1) != mask) |
#             (numpy.roll(mask, -1, axis=2) != mask) |
#             (numpy.roll(mask, +1, axis=2) != mask))

    return sparse.diags(jmask.astype(int)) @ full

