"""
Operators and helper functions for waveguides with unchanging cross-section.

The propagation direction is chosen to be along the z axis, and all fields
are given an implicit z-dependence of the form `exp(-1 * wavenumber * z)`.

As the z-dependence is known, all the functions in this file assume a 2D grid
 (i.e. `dxes = [[[dx_e[0], dx_e[1], ...], [dy_e[0], ...]], [[dx_h[0], ...], [dy_h[0], ...]]]`).


===============

Consider Maxwell's equations in continuous space, in the frequency domain. Assuming
a structure with some (x, y) cross-section extending uniformly into the z dimension,
with a diagonal $\\epsilon$ tensor, we have

$$
\\begin{aligned}
\\nabla \\times \\vec{E}(x, y, z) &= -\\imath \\omega \\mu \\vec{H} \\\\
\\nabla \\times \\vec{H}(x, y, z) &=  \\imath \\omega \\epsilon \\vec{E} \\\\
\\vec{E}(x,y,z) = (\\vec{E}_t(x, y) + E_z(x, y)\\vec{z}) e^{-\\gamma z} \\\\
\\vec{H}(x,y,z) = (\\vec{H}_t(x, y) + H_z(x, y)\\vec{z}) e^{-\\gamma z} \\\\
\\end{aligned}
$$

Expanding the first two equations into vector components, we get

$$
\\begin{aligned}
-\\imath \\omega \\mu_{xx} H_x &= \\partial_y E_z - \\partial_z E_y \\\\
-\\imath \\omega \\mu_{yy} H_y &= \\partial_z E_x - \\partial_x E_z \\\\
-\\imath \\omega \\mu_{zz} H_z &= \\partial_x E_y - \\partial_y E_x \\\\
\\imath \\omega \\epsilon_{xx} E_x &= \\partial_y H_z - \\partial_z H_y \\\\
\\imath \\omega \\epsilon_{yy} E_y &= \\partial_z H_x - \\partial_x H_z \\\\
\\imath \\omega \\epsilon_{zz} E_z &= \\partial_x H_y - \\partial_y H_x \\\\
\\end{aligned}
$$

Substituting in our expressions for $\\vec{E}$, $\\vec{H}$ and discretizing:

$$
\\begin{aligned}
-\\imath \\omega \\mu_{xx} H_x &= \\tilde{\\partial}_y E_z + \\gamma E_y \\\\
-\\imath \\omega \\mu_{yy} H_y &= -\\gamma E_x - \\tilde{\\partial}_x E_z \\\\
-\\imath \\omega \\mu_{zz} H_z &= \\tilde{\\partial}_x E_y - \\tilde{\\partial}_y E_x \\\\
\\imath \\omega \\epsilon_{xx} E_x &= \\hat{\\partial}_y H_z + \\gamma H_y \\\\
\\imath \\omega \\epsilon_{yy} E_y &= -\\gamma H_x - \\hat{\\partial}_x H_z \\\\
\\imath \\omega \\epsilon_{zz} E_z &= \\hat{\\partial}_x H_y - \\hat{\\partial}_y H_x \\\\
\\end{aligned}
$$

Rewrite the last three equations as
$$
\\begin{aligned}
\\gamma H_y &=  \\imath \\omega \\epsilon_{xx} E_x - \\hat{\\partial}_y H_z \\\\
\\gamma H_x &= -\\imath \\omega \\epsilon_{yy} E_y - \\hat{\\partial}_x H_z \\\\
\\imath \\omega E_z &= \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x H_y - \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y H_x \\\\
\\end{aligned}
$$

Now apply $\\gamma \\tilde{\\partial}_x$ to the last equation,
then substitute in for $\\gamma H_x$ and $\\gamma H_y$:

$$
\\begin{aligned}
\\gamma \\tilde{\\partial}_x \\imath \\omega E_z &= \\gamma \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x H_y
                                                - \\gamma \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y H_x \\\\
        &= \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x ( \\imath \\omega \\epsilon_{xx} E_x - \\hat{\\partial}_y H_z)
         - \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (-\\imath \\omega \\epsilon_{yy} E_y - \\hat{\\partial}_x H_z)  \\\\
        &= \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x ( \\imath \\omega \\epsilon_{xx} E_x)
         - \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (-\\imath \\omega \\epsilon_{yy} E_y)  \\\\
\\gamma \\tilde{\\partial}_x E_z &= \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x (\\epsilon_{xx} E_x)
                                  + \\tilde{\\partial}_x \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (\\epsilon_{yy} E_y) \\\\
\\end{aligned}
$$

With a similar approach (but using $\\gamma \\tilde{\\partial}_y$ instead), we can get

$$
\\begin{aligned}
\\gamma \\tilde{\\partial}_y E_z &= \\tilde{\\partial}_y \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x (\\epsilon_{xx} E_x)
                                  + \\tilde{\\partial}_y \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (\\epsilon_{yy} E_y) \\\\
\\end{aligned}
$$

We can combine this equation for $\\gamma \\tilde{\\partial}_y E_z$ with
the unused $\\imath \\omega \\mu_{xx} H_x$ and $\\imath \\omega \\mu_{yy} H_y$ equations to get

$$
\\begin{aligned}
-\\imath \\omega \\mu_{xx} \\gamma H_x &=  \\gamma^2 E_y + \\gamma \\tilde{\\partial}_y E_z \\\\
-\\imath \\omega \\mu_{xx} \\gamma H_x &=  \\gamma^2 E_y + \\tilde{\\partial}_y (
                                      \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x (\\epsilon_{xx} E_x)
                                    + \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (\\epsilon_{yy} E_y)
                                    )\\\\
\\end{aligned}
$$

and

$$
\\begin{aligned}
-\\imath \\omega \\mu_{yy} \\gamma H_y &= -\\gamma^2 E_x - \\gamma \\tilde{\\partial}_x E_z \\\\
-\\imath \\omega \\mu_{yy} \\gamma H_y &= -\\gamma^2 E_x - \\tilde{\\partial}_x (
                                      \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x (\\epsilon_{xx} E_x)
                                    + \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (\\epsilon_{yy} E_y)
                                    )\\\\
\\end{aligned}
$$

However, based on our rewritten equation for $\\gamma H_x$ and the so-far unused
equation for $\\imath \\omega \\mu_{zz} H_z$ we can also write

$$
\\begin{aligned}
-\\imath \\omega \\mu_{xx} (\\gamma H_x) &= -\\imath \\omega \\mu_{xx} (-\\imath \\omega \\epsilon_{yy} E_y - \\hat{\\partial}_x H_z) \\\\
                                         &= -\\omega^2 \\mu_{xx} \\epsilon_{yy} E_y
                                            +\\imath \\omega \\mu_{xx} \\hat{\\partial}_x (
                                                \\frac{1}{-\\imath \\omega \\mu_{zz}} (\\tilde{\\partial}_x E_y - \\tilde{\\partial}_y E_x)) \\\\
                                         &= -\\omega^2 \\mu_{xx} \\epsilon_{yy} E_y
                                            -\\mu_{xx} \\hat{\\partial}_x \\frac{1}{\\mu_{zz}} (\\tilde{\\partial}_x E_y - \\tilde{\\partial}_y E_x) \\\\
\\end{aligned}
$$

and, similarly,

$$
\\begin{aligned}
-\\imath \\omega \\mu_{yy} (\\gamma H_y) &= \\omega^2 \\mu_{yy} \\epsilon_{xx} E_x
                                           +\\mu_{yy} \\hat{\\partial}_y \\frac{1}{\\mu_{zz}} (\\tilde{\\partial}_x E_y - \\tilde{\\partial}_y E_x) \\\\
\\end{aligned}
$$

By combining both pairs of expressions, we get

$$
\\begin{aligned}
-\\gamma^2 E_x - \\tilde{\\partial}_x (
    \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x (\\epsilon_{xx} E_x)
  + \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (\\epsilon_{yy} E_y)
    ) &= \\omega^2 \\mu_{yy} \\epsilon_{xx} E_x
        +\\mu_{yy} \\hat{\\partial}_y \\frac{1}{\\mu_{zz}} (\\tilde{\\partial}_x E_y - \\tilde{\\partial}_y E_x) \\\\
\\gamma^2 E_y + \\tilde{\\partial}_y (
    \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_x (\\epsilon_{xx} E_x)
  + \\frac{1}{\\epsilon_{zz}} \\hat{\\partial}_y (\\epsilon_{yy} E_y)
    ) &= -\\omega^2 \\mu_{xx} \\epsilon_{yy} E_y
         -\\mu_{xx} \\hat{\\partial}_x \\frac{1}{\\mu_{zz}} (\\tilde{\\partial}_x E_y - \\tilde{\\partial}_y E_x) \\\\
\\end{aligned}
$$

Using these, we can construct the eigenvalue problem

$$
\\beta^2 \\begin{bmatrix} E_x \\\\
                          E_y \\end{bmatrix} =
    (\\omega^2 \\begin{bmatrix} \\mu_{yy} \\epsilon_{xx} & 0 \\\\
                                                       0 & \\mu_{xx} \\epsilon_{yy} \\end{bmatrix} +
                 \\begin{bmatrix} -\\mu_{yy} \\hat{\\partial}_y \\\\
                                   \\mu_{xx} \\hat{\\partial}_x \\end{bmatrix} \\mu_{zz}^{-1}
                 \\begin{bmatrix} -\\tilde{\\partial}_y & \\tilde{\\partial}_x \\end{bmatrix} +
      \\begin{bmatrix} \\tilde{\\partial}_x \\\\
                       \\tilde{\\partial}_y \\end{bmatrix} \\epsilon_{zz}^{-1}
                 \\begin{bmatrix} \\hat{\\partial}_x \\epsilon_{xx} & \\hat{\\partial}_y \\epsilon_{yy} \\end{bmatrix})
    \\begin{bmatrix} E_x \\\\
                     E_y \\end{bmatrix}
$$

where $\\gamma = \\imath\\beta$. In the literature, $\\beta$ is usually used to denote
the lossless/real part of the propagation constant, but in `meanas` it is allowed to
be complex.

An equivalent eigenvalue problem can be formed using the $H_x$ and $H_y$ fields, if those are more convenient.

Note that $E_z$ was never discretized, so $\\gamma$ and $\\beta$ will need adjustment
to account for numerical dispersion if the result is introduced into a space with a discretized z-axis.


"""
# TODO update module docs

from typing import List, Tuple, Optional, Any
import numpy
from numpy.typing import NDArray, ArrayLike
from numpy.linalg import norm
import scipy.sparse as sparse       # type: ignore

from ..fdmath.operators import deriv_forward, deriv_back, cross
from ..fdmath import unvec, dx_lists_t, vfdfield_t, vcfdfield_t
from ..eigensolvers import signed_eigensolve, rayleigh_quotient_iteration


__author__ = 'Jan Petykiewicz'


def operator_e(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Waveguide operator of the form

        omega**2 * mu * epsilon +
        mu * [[-Dy], [Dx]] / mu * [-Dy, Dx] +
        [[Dx], [Dy]] / epsilon * [Dx, Dy] * epsilon

    for use with a field vector of the form `cat([E_x, E_y])`.

    More precisely, the operator is

    $$
    \\omega^2 \\begin{bmatrix} \\mu_{yy} \\epsilon_{xx} & 0 \\\\
                                                       0 & \\mu_{xx} \\epsilon_{yy} \\end{bmatrix} +
                 \\begin{bmatrix} -\\mu_{yy} \\hat{\\partial}_y \\\\
                                   \\mu_{xx} \\hat{\\partial}_x \\end{bmatrix} \\mu_{zz}^{-1}
                 \\begin{bmatrix} -\\tilde{\\partial}_y & \\tilde{\\partial}_x \\end{bmatrix} +
      \\begin{bmatrix} \\tilde{\\partial}_x \\\\
                       \\tilde{\\partial}_y \\end{bmatrix} \\epsilon_{zz}^{-1}
                 \\begin{bmatrix} \\hat{\\partial}_x \\epsilon_{xx} & \\hat{\\partial}_y \\epsilon_{yy} \\end{bmatrix}
    $$

    $\\tilde{\\partial}_x$ and $\\hat{\\partial}_x$ are the forward and backward derivatives along x,
    and each $\\epsilon_{xx}$, $\\mu_{yy}$, etc. is a diagonal matrix containing the vectorized material
    property distribution.

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
    if mu is None:
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


def operator_h(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        ) -> sparse.spmatrix:
    """
    Waveguide operator of the form

        omega**2 * epsilon * mu +
        epsilon * [[-Dy], [Dx]] / epsilon * [-Dy, Dx] +
        [[Dx], [Dy]] / mu * [Dx, Dy] * mu

    for use with a field vector of the form `cat([H_x, H_y])`.

    More precisely, the operator is

    $$
    \\omega^2 \\begin{bmatrix} \\epsilon_{yy} \\mu_{xx} & 0 \\\\
                                                      0 & \\epsilon_{xx} \\mu_{yy} \\end{bmatrix} +
                 \\begin{bmatrix} -\\epsilon_{yy} \\tilde{\\partial}_y \\\\
                                   \\epsilon_{xx} \\tilde{\\partial}_x \\end{bmatrix} \\epsilon_{zz}^{-1}
                 \\begin{bmatrix} -\\hat{\\partial}_y & \\hat{\\partial}_x \\end{bmatrix} +
      \\begin{bmatrix} \\hat{\\partial}_x \\\\
                       \\hat{\\partial}_y \\end{bmatrix} \\mu_{zz}^{-1}
                 \\begin{bmatrix} \\tilde{\\partial}_x \\mu_{xx} & \\tilde{\\partial}_y \\mu_{yy} \\end{bmatrix}
    $$

    $\\tilde{\\partial}_x$ and $\\hat{\\partial}_x$ are the forward and backward derivatives along x,
    and each $\\epsilon_{xx}$, $\\mu_{yy}$, etc. is a diagonal matrix containing the vectorized material
    property distribution.

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
    if mu is None:
        mu = numpy.ones_like(epsilon)

    Dfx, Dfy = deriv_forward(dxes[0])
    Dbx, Dby = deriv_back(dxes[1])

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


def normalized_fields_e(
        e_xy: ArrayLike,
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        prop_phase: float = 0,
        ) -> Tuple[vcfdfield_t, vcfdfield_t]:
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


def normalized_fields_h(
        h_xy: ArrayLike,
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        prop_phase: float = 0,
        ) -> Tuple[vcfdfield_t, vcfdfield_t]:
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


def _normalized_fields(
        e: vcfdfield_t,
        h: vcfdfield_t,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
        prop_phase: float = 0,
        ) -> Tuple[vcfdfield_t, vcfdfield_t]:
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
    sign = numpy.sign(E_weighted[:,
                                 :max(shape[0] // 2, 1),
                                 :max(shape[1] // 2, 1)].real.sum())

    norm_factor = sign * norm_amplitude * numpy.exp(1j * norm_angle)

    e *= norm_factor
    h *= norm_factor

    return e, h


def exy2h(
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None
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


def hxy2e(
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None
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


def hxy2h(
        wavenumber: complex,
        dxes: dx_lists_t,
        mu: Optional[vfdfield_t] = None
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

    if mu is not None:
        mu_parts = numpy.split(mu, 3)
        mu_xy = sparse.diags(numpy.hstack((mu_parts[0], mu_parts[1])))
        mu_z_inv = sparse.diags(1 / mu_parts[2])

        hxy2hz = mu_z_inv @ hxy2hz @ mu_xy

    n_pts = dxes[1][0].size * dxes[1][1].size
    op = sparse.vstack((sparse.eye(2 * n_pts),
                        hxy2hz))
    return op


def exy2e(
        wavenumber: complex,
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

    if epsilon is not None:
        epsilon_parts = numpy.split(epsilon, 3)
        epsilon_xy = sparse.diags(numpy.hstack((epsilon_parts[0], epsilon_parts[1])))
        epsilon_z_inv = sparse.diags(1 / epsilon_parts[2])

        exy2ez = epsilon_z_inv @ exy2ez @ epsilon_xy

    n_pts = dxes[0][0].size * dxes[0][1].size
    op = sparse.vstack((sparse.eye(2 * n_pts),
                        exy2ez))
    return op


def e2h(
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        mu: Optional[vfdfield_t] = None
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
    if mu is not None:
        op = sparse.diags(1 / mu) @ op
    return op


def h2e(
        wavenumber: complex,
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

    Returns:
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

    Returns:
        Sparse matrix representation of the operator.
    """
    n = 1
    for d in dxes[1]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dbx, Dby = deriv_back(dxes[1])
    return cross([Dbx, Dby, Bz])


def h_err(
        h: vcfdfield_t,
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None
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

    if mu is None:
        op = ce @ eps_inv @ ch @ h - omega ** 2 * h
    else:
        op = ce @ eps_inv @ ch @ h - omega ** 2 * (mu * h)

    return norm(op) / norm(h)


def e_err(
        e: vcfdfield_t,
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: Optional[vfdfield_t] = None,
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

    if mu is None:
        op = ch @ ce @ e - omega ** 2 * (epsilon * e)
    else:
        mu_inv = sparse.diags(1 / mu)
        op = ch @ mu_inv @ ce @ e - omega ** 2 * (epsilon * e)

    return norm(op) / norm(e)


def solve_modes(
        mode_numbers: List[int],
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        mu: vfdfield_t = None,
        mode_margin: int = 2,
        ) -> Tuple[NDArray[numpy.float64], NDArray[numpy.complex128]]:
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
        e_xys: list of vfdfield_t specifying fields
        wavenumbers: list of wavenumbers
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


def solve_mode(
        mode_number: int,
        *args: Any,
        **kwargs: Any,
        ) -> Tuple[vcfdfield_t, complex]:
    """
    Wrapper around `solve_modes()` that solves for a single mode.

    Args:
       mode_number: 0-indexed mode number to solve for
       *args: passed to `solve_modes()`
       **kwargs: passed to `solve_modes()`

    Returns:
        (e_xy, wavenumber)
    """
    kwargs['mode_numbers'] = [mode_number]
    e_xys, wavenumbers = solve_modes(*args, **kwargs)
    return e_xys[:, 0], wavenumbers[0]
