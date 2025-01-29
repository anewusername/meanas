r"""
Operators and helper functions for cylindrical waveguides with unchanging cross-section.

Waveguide operator is derived according to 10.1364/OL.33.001848.
The curl equations in the complex coordinate system become

$$
\begin{aligned}
-\imath \omega \mu_{xx} H_x &= \tilde{\partial}_y E_z + \imath \beta frac{E_y}{\tilde{t}_x} \\
-\imath \omega \mu_{yy} H_y &= -\imath \beta E_x - \frac{1}{\hat{t}_x} \tilde{\partial}_x \tilde{t}_x E_z \\
-\imath \omega \mu_{zz} H_z &= \tilde{\partial}_x E_y - \tilde{\partial}_y E_x \\
\imath \omega \epsilon_{xx} E_x &= \hat{\partial}_y H_z + \imath \beta \frac{H_y}{\hat{T}} \\
\imath \omega \epsilon_{yy} E_y &= -\imath \beta H_x - \{1}{\tilde{t}_x} \hat{\partial}_x \hat{t}_x} H_z \\
\imath \omega \epsilon_{zz} E_z &= \hat{\partial}_x H_y - \hat{\partial}_y H_x \\
\end{aligned}
$$

where $t_x = 1 + \frac{\Delta_{x, m}}{R_0}$ is the grid spacing adjusted by the nominal radius $R0$.

Rewrite the last three equations as

$$
\begin{aligned}
\imath \beta H_y &=  \imath \omega \hat{t}_x \epsilon_{xx} E_x - \hat{t}_x \hat{\partial}_y H_z \\
\imath \beta H_x &= -\imath \omega \hat{t}_x \epsilon_{yy} E_y - \hat{t}_x \hat{\partial}_x H_z \\
\imath \omega E_z &= \frac{1}{\epsilon_{zz}} \hat{\partial}_x H_y - \frac{1}{\epsilon_{zz}} \hat{\partial}_y H_x \\
\end{aligned}
$$

The derivation then follows the same steps as the straight waveguide, leading to the eigenvalue problem

$$
\beta^2 \begin{bmatrix} E_x \\
                        E_y \end{bmatrix} =
    (\omega^2 \begin{bmatrix} T_b T_b \mu_{yy} \epsilon_{xx} & 0 \\
                                                            0 & T_a T_a \mu_{xx} \epsilon_{yy} \end{bmatrix} +
              \begin{bmatrix} -T_b \mu_{yy} \hat{\partial}_y \\
                               T_a \mu_{xx} \hat{\partial}_x \end{bmatrix} T_b \mu_{zz}^{-1}
              \begin{bmatrix} -\tilde{\partial}_y & \tilde{\partial}_x \end{bmatrix} +
      \begin{bmatrix} \tilde{\partial}_x \\
                      \tilde{\partial}_y \end{bmatrix} T_a \epsilon_{zz}^{-1}
                 \begin{bmatrix} \hat{\partial}_x T_b \epsilon_{xx} & \hat{\partial}_y T_a \epsilon_{yy} \end{bmatrix})
    \begin{bmatrix} E_x \\
                    E_y \end{bmatrix}
$$

which resembles the straight waveguide eigenproblem with additonal $T_a$ and $T_b$ terms. These
are diagonal matrices containing the $t_x$ values:

$$
\begin{aligned}
T_a &=  1 + \frac{\Delta_{x, m               }}{R_0}
T_b &=  1 + \frac{\Delta_{x, m + \frac{1}{2} }}{R_0}
\end{aligned}


TODO: consider 10.1364/OE.20.021583 for an alternate approach
$$

As in the straight waveguide case, all the functions in this file assume a 2D grid
 (i.e. `dxes = [[[dr_e_0, dx_e_1, ...], [dy_e_0, ...]], [[dr_h_0, ...], [dy_h_0, ...]]]`).
"""
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
        omega: float,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        rmin: float,
        ) -> sparse.spmatrix:
    r"""
    Cylindrical coordinate waveguide operator of the form

    $$
        (\omega^2 \begin{bmatrix} T_b T_b \mu_{yy} \epsilon_{xx} & 0 \\
                                                                0 & T_a T_a \mu_{xx} \epsilon_{yy} \end{bmatrix} +
                  \begin{bmatrix} -T_b \mu_{yy} \hat{\partial}_y \\
                                   T_a \mu_{xx} \hat{\partial}_x \end{bmatrix} T_b \mu_{zz}^{-1}
                  \begin{bmatrix} -\tilde{\partial}_y & \tilde{\partial}_x \end{bmatrix} +
          \begin{bmatrix} \tilde{\partial}_x \\
                          \tilde{\partial}_y \end{bmatrix} T_a \epsilon_{zz}^{-1}
                     \begin{bmatrix} \hat{\partial}_x T_b \epsilon_{xx} & \hat{\partial}_y T_a \epsilon_{yy} \end{bmatrix})
        \begin{bmatrix} E_x \\
                        E_y \end{bmatrix}
    $$

    for use with a field vector of the form `[E_r, E_y]`.

    This operator can be used to form an eigenvalue problem of the form
        A @ [E_r, E_y] = wavenumber**2 * [E_r, E_y]

    which can then be solved for the eigenmodes of the system
    (an `exp(-i * wavenumber * theta)` theta-dependence is assumed for the fields).

    (NOTE: See module docs and 10.1364/OL.33.001848)

    Args:
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        epsilon: Vectorized dielectric constant grid
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')

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
    op = sq0 + lin0 + lin1
    return op


def solve_modes(
        mode_numbers: Sequence[int],
        omega: float,
        dxes: dx_lists_t,
        epsilon: vfdfield_t,
        rmin: float,
        mode_margin: int = 2,
        ) -> tuple[vcfdfield_t, NDArray[numpy.complex128]]:
    """
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
        e_xys: Vectorized mode fields with shape (num_modes, 2 * x *y)
        angular_wavenumbers: Wavenumbers assuming fields have theta-dependence of
            `exp(-i * angular_wavenumber * theta)`. They should satisfy
            `operator_e() @ e_xy == (angular_wavenumber / rmin) ** 2 * e_xy`
        epsilon: Vectorized dielectric constant grid with shape (3, x, y)
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')

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


def exy2h(
        angular_wavenumber: complex,
        omega: float,
        dxes: dx_lists_t,
        rmin: float,
        epsilon: vfdfield_t,
        mu: vfdfield_t | None = None
        ) -> sparse.spmatrix:
    """
    Operator which transforms the vector `e_xy` containing the vectorized E_x and E_y fields,
     into a vectorized H containing all three H components

    Args:
        angular_wavenumber: Wavenumber assuming fields have theta-dependence of
            `exp(-i * angular_wavenumber * theta)`. It should satisfy
            `operator_e() @ e_xy == (angular_wavenumber / rmin) ** 2 * e_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representing the operator.
    """
    e2hop = e2h(angular_wavenumber=angular_wavenumber, omega=omega, dxes=dxes, rmin=rmin, mu=mu)
    return e2hop @ exy2e(angular_wavenumber=angular_wavenumber, omega=omega, dxes=dxes, rmin=rmin, epsilon=epsilon)


def exy2e(
        angular_wavenumber: complex,
        omega: float,
        dxes: dx_lists_t,
        rmin: float,
        epsilon: vfdfield_t,
        ) -> sparse.spmatrix:
    """
    Operator which transforms the vector `e_xy` containing the vectorized E_x and E_y fields,
     into a vectorized E containing all three E components

    Unlike the straight waveguide case, the H_z components do not cancel and must be calculated
    from E_x and E_y in order to then calculate E_z.

    Args:
        angular_wavenumber: Wavenumber assuming fields have theta-dependence of
            `exp(-i * angular_wavenumber * theta)`. It should satisfy
            `operator_e() @ e_xy == (angular_wavenumber / rmin) ** 2 * e_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')
        epsilon: Vectorized dielectric constant grid

    Returns:
        Sparse matrix representing the operator.
    """
    Dfx, Dfy = deriv_forward(dxes[0])
    Dbx, Dby = deriv_back(dxes[1])
    wavenumber = angular_wavenumber / rmin

    Ta, Tb = dxes2T(dxes=dxes, rmin=rmin)
    Tai = sparse.diags_array(1 / Ta.diagonal())
    Tbi = sparse.diags_array(1 / Tb.diagonal())

    epsilon_parts = numpy.split(epsilon, 3)
    epsilon_x, epsilon_y = (sparse.diags_array(epsi) for epsi in epsilon_parts[:2])
    epsilon_z_inv = sparse.diags_array(1 / epsilon_parts[2])

    n_pts = dxes[0][0].size * dxes[0][1].size
    zeros = sparse.coo_array((n_pts, n_pts))
    keep_x = sparse.block_array([[sparse.eye_array(n_pts), None], [None, zeros]])
    keep_y = sparse.block_array([[zeros, None], [None, sparse.eye_array(n_pts)]])

    mu_z = numpy.ones(n_pts)
    mu_z_inv = sparse.diags_array(1 / mu_z)
    exy2hz = 1 / (-1j * omega) * mu_z_inv @ sparse.hstack((Dfy, -Dfx))
    hxy2ez = 1 / (1j * omega) * epsilon_z_inv @ sparse.hstack((Dby, -Dbx))

    exy2hy = Tb / (1j * wavenumber) @ (-1j * omega * sparse.hstack((epsilon_x, zeros)) - Dby @ exy2hz)
    exy2hx = Tb / (1j * wavenumber) @ ( 1j * omega * sparse.hstack((zeros, epsilon_y)) - Tai @ Dbx @ Tb @ exy2hz)

    exy2ez = hxy2ez @ sparse.vstack((exy2hx, exy2hy))

    op = sparse.vstack((sparse.eye_array(2 * n_pts),
                        exy2ez))
    return op


def e2h(
        angular_wavenumber: complex,
        omega: float,
        dxes: dx_lists_t,
        rmin: float,
        mu: vfdfield_t | None = None
        ) -> sparse.spmatrix:
    r"""
    Returns an operator which, when applied to a vectorized E eigenfield, produces
     the vectorized H eigenfield.

    This operator is created directly from the initial coordinate-transformed equations:
    $$
    \begin{aligned}
    \imath \omega \epsilon_{xx} E_x &= \hat{\partial}_y H_z + \imath \beta \frac{H_y}{\hat{T}} \\
    \imath \omega \epsilon_{yy} E_y &= -\imath \beta H_x - \{1}{\tilde{t}_x} \hat{\partial}_x \hat{t}_x} H_z \\
    \imath \omega \epsilon_{zz} E_z &= \hat{\partial}_x H_y - \hat{\partial}_y H_x \\
    \end{aligned}
    $$

    Args:
        angular_wavenumber: Wavenumber assuming fields have theta-dependence of
            `exp(-i * angular_wavenumber * theta)`. It should satisfy
            `operator_e() @ e_xy == (angular_wavenumber / rmin) ** 2 * e_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')
        mu: Vectorized magnetic permeability grid (default 1 everywhere)

    Returns:
        Sparse matrix representation of the operator.
    """
    Dfx, Dfy = deriv_forward(dxes[0])
    Ta, Tb = dxes2T(dxes=dxes, rmin=rmin)
    Tai = sparse.diags_array(1 / Ta.diagonal())
    Tbi = sparse.diags_array(1 / Tb.diagonal())

    jB = 1j * angular_wavenumber / rmin
    op = sparse.block_array([[    None, -jB * Tai,           -Dfy],
                             [jB * Tbi,      None, Tbi @ Dfx @ Ta],
                             [     Dfy,      -Dfx,           None]]) / (-1j * omega)
    if mu is not None:
        op = sparse.diags_array(1 / mu) @ op
    return op


def dxes2T(
        dxes: dx_lists_t,
        rmin: float,
        ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""
    Returns the $T_a$ and $T_b$ diagonal matrices which are used to apply the cylindrical
      coordinate transformation in various operators.

    Args:
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')

    Returns:
        Sparse matrix representations of the operators Ta and Tb
    """
    ra = rmin + numpy.cumsum(dxes[0][0])                      # Radius at Ey points
    rb = rmin + dxes[0][0] / 2.0 + numpy.cumsum(dxes[1][0])   # Radius at Ex points
    ta = ra / rmin
    tb = rb / rmin

    Ta = sparse.diags_array(vec(ta[:, None].repeat(dxes[0][1].size, axis=1)))
    Tb = sparse.diags_array(vec(tb[:, None].repeat(dxes[1][1].size, axis=1)))
    return Ta, Tb


def normalized_fields_e(
        e_xy: ArrayLike,
        angular_wavenumber: complex,
        omega: float,
        dxes: dx_lists_t,
        rmin: float,
        epsilon: vfdfield_t,
        mu: vfdfield_t | None = None,
        prop_phase: float = 0,
        ) -> tuple[vcfdfield_t, vcfdfield_t]:
    """
    Given a vector `e_xy` containing the vectorized E_x and E_y fields,
     returns normalized, vectorized E and H fields for the system.

    Args:
        e_xy: Vector containing E_x and E_y fields
        angular_wavenumber: Wavenumber assuming fields have theta-dependence of
            `exp(-i * angular_wavenumber * theta)`. It should satisfy
            `operator_e() @ e_xy == (angular_wavenumber / rmin) ** 2 * e_xy`
        omega: The angular frequency of the system
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types` (2D)
        rmin: Radius at the left edge of the simulation domain (at minimum 'x')
        epsilon: Vectorized dielectric constant grid
        mu: Vectorized magnetic permeability grid (default 1 everywhere)
        prop_phase: Phase shift `(dz * corrected_wavenumber)` over 1 cell in propagation direction.
                    Default 0 (continuous propagation direction, i.e. dz->0).

    Returns:
        `(e, h)`, where each field is vectorized, normalized,
        and contains all three vector components.
    """
    e = exy2e(angular_wavenumber=angular_wavenumber, omega=omega, dxes=dxes, rmin=rmin, epsilon=epsilon) @ e_xy
    h = exy2h(angular_wavenumber=angular_wavenumber, omega=omega, dxes=dxes, rmin=rmin, epsilon=epsilon, mu=mu) @ e_xy
    e_norm, h_norm = _normalized_fields(e=e, h=h, omega=omega, dxes=dxes, rmin=rmin, epsilon=epsilon,
                                        mu=mu, prop_phase=prop_phase)
    return e_norm, h_norm


def _normalized_fields(
        e: vcfdfield_t,
        h: vcfdfield_t,
        omega: complex,
        dxes: dx_lists_t,
        rmin: float,
        epsilon: vfdfield_t,
        mu: vfdfield_t | None = None,
        prop_phase: float = 0,
        ) -> tuple[vcfdfield_t, vcfdfield_t]:
    h *= -1
    # TODO documentation for normalized_fields
    shape = [s.size for s in dxes[0]]
    dxes_real = [[numpy.real(d) for d in numpy.meshgrid(*dxes[v], indexing='ij')] for v in (0, 1)]

    # Find time-averaged Sz and normalize to it
    # H phase is adjusted by a half-cell forward shift for Yee cell, and 1-cell reverse shift for Poynting
    phase = numpy.exp(-1j * -prop_phase / 2)
    Sz_tavg = waveguide_2d.inner_product(e, h, dxes=dxes, prop_phase=prop_phase, conj_h=True).real   # Note, using linear poynting vector
    assert Sz_tavg > 0, f'Found a mode propagating in the wrong direction! {Sz_tavg=}'

    energy = numpy.real(epsilon * e.conj() * e)

    norm_amplitude = 1 / numpy.sqrt(Sz_tavg)
    norm_angle = -numpy.angle(e[energy.argmax()])       # Will randomly add a negative sign when mode is symmetric

    # Try to break symmetry to assign a consistent sign [experimental]
    E_weighted = unvec(e * energy * numpy.exp(1j * norm_angle), shape)
    sign = numpy.sign(E_weighted[:,
                                 :max(shape[0] // 2, 1),
                                 :max(shape[1] // 2, 1)].real.sum())
    assert sign != 0

    norm_factor = sign * norm_amplitude * numpy.exp(1j * norm_angle)

    print('\nAAA\n', waveguide_2d.inner_product(e, h, dxes, prop_phase=prop_phase))
    e *= norm_factor
    h *= norm_factor
    print(f'{sign=} {norm_amplitude=} {norm_angle=} {prop_phase=}')
    print(waveguide_2d.inner_product(e, h, dxes, prop_phase=prop_phase))

    return e, h
