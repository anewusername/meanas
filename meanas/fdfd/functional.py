"""
Functional versions of many FDFD operators. These can be useful for performing
 FDFD calculations without needing to construct large matrices in memory.

The functions generated here expect `cfdfield_t` inputs with shape (3, X, Y, Z),
e.g. E = [E_x, E_y, E_z] where each (complex) component has shape (X, Y, Z)
"""
from typing import Callable, Tuple, Optional
import numpy

from ..fdmath import dx_lists_t, fdfield_t, cfdfield_t, cfdfield_updater_t
from ..fdmath.functional import curl_forward, curl_back


__author__ = 'Jan Petykiewicz'


def e_full(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: fdfield_t,
        mu: Optional[fdfield_t] = None
        ) -> cfdfield_updater_t:
    """
    Wave operator for use with E-field. See `operators.e_full` for details.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Dielectric constant
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        Function `f` implementing the wave operator
        `f(E)` -> `-i * omega * J`
    """
    ch = curl_back(dxes[1])
    ce = curl_forward(dxes[0])

    def op_1(e: cfdfield_t) -> cfdfield_t:
        curls = ch(ce(e))
        return curls - omega ** 2 * epsilon * e

    def op_mu(e: cfdfield_t) -> cfdfield_t:
        curls = ch(mu * ce(e))          # type: ignore   # mu = None ok because we don't return the function
        return curls - omega ** 2 * epsilon * e

    if mu is None:
        return op_1
    else:
        return op_mu


def eh_full(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: fdfield_t,
        mu: fdfield_t = None
        ) -> Callable[[cfdfield_t, cfdfield_t], Tuple[cfdfield_t, cfdfield_t]]:
    """
    Wave operator for full (both E and H) field representation.
    See `operators.eh_full`.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Dielectric constant
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        Function `f` implementing the wave operator
        `f(E, H)` -> `(J, -M)`
    """
    ch = curl_back(dxes[1])
    ce = curl_forward(dxes[0])

    def op_1(e: cfdfield_t, h: cfdfield_t) -> Tuple[cfdfield_t, cfdfield_t]:
        return (ch(h) - 1j * omega * epsilon * e,
                ce(e) + 1j * omega * h)

    def op_mu(e: cfdfield_t, h: cfdfield_t) -> Tuple[cfdfield_t, cfdfield_t]:
        return (ch(h) - 1j * omega * epsilon * e,
                ce(e) + 1j * omega * mu * h)            # type: ignore   # mu=None ok

    if mu is None:
        return op_1
    else:
        return op_mu


def e2h(
        omega: complex,
        dxes: dx_lists_t,
        mu: Optional[fdfield_t] = None,
        ) -> cfdfield_updater_t:
    """
    Utility operator for converting the `E` field into the `H` field.
    For use with `e_full` -- assumes that there is no magnetic current `M`.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        Function `f` for converting `E` to `H`,
        `f(E)` -> `H`
    """
    ce = curl_forward(dxes[0])

    def e2h_1_1(e: cfdfield_t) -> cfdfield_t:
        return ce(e) / (-1j * omega)

    def e2h_mu(e: cfdfield_t) -> cfdfield_t:
        return ce(e) / (-1j * omega * mu)       # type: ignore   # mu=None ok

    if mu is None:
        return e2h_1_1
    else:
        return e2h_mu


def m2j(
        omega: complex,
        dxes: dx_lists_t,
        mu: Optional[fdfield_t] = None,
        ) -> cfdfield_updater_t:
    """
    Utility operator for converting magnetic current `M` distribution
    into equivalent electric current distribution `J`.
    For use with e.g. `e_full`.

    Args:
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        Function `f` for converting `M` to `J`,
        `f(M)` -> `J`
    """
    ch = curl_back(dxes[1])

    def m2j_mu(m: cfdfield_t) -> cfdfield_t:
        J = ch(m / mu) / (-1j * omega)          # type: ignore  # mu=None ok
        return J

    def m2j_1(m: cfdfield_t) -> cfdfield_t:
        J = ch(m) / (-1j * omega)
        return J

    if mu is None:
        return m2j_1
    else:
        return m2j_mu


def e_tfsf_source(
        TF_region: fdfield_t,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: fdfield_t,
        mu: Optional[fdfield_t] = None,
        ) -> cfdfield_updater_t:
    """
    Operator that turns an E-field distribution into a total-field/scattered-field
    (TFSF) source.

    Args:
        TF_region: mask which is set to 1 in the total-field region, and 0 elsewhere
                   (i.e. in the scattered-field region).
                   Should have the same shape as the simulation grid, e.g. `epsilon[0].shape`.
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        epsilon: Dielectric constant distribution
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        Function `f` which takes an E field and returns a current distribution,
        `f(E)` -> `J`
    """
    # TODO documentation
    A = e_full(omega, dxes, epsilon, mu)

    def op(e: cfdfield_t) -> cfdfield_t:
        neg_iwj = A(TF_region * e) - TF_region * A(e)
        return neg_iwj / (-1j * omega)
    return op


def poynting_e_cross_h(dxes: dx_lists_t) -> Callable[[cfdfield_t, cfdfield_t], cfdfield_t]:
    """
    Generates a function that takes the single-frequency `E` and `H` fields
    and calculates the cross product `E` x `H` = $E \\times H$ as required
    for the Poynting vector, $S = E \\times H$

    Note:
        This function also shifts the input `E` field by one cell as required
        for computing the Poynting cross product (see `meanas.fdfd` module docs).

    Note:
        If `E` and `H` are peak amplitudes as assumed elsewhere in this code,
        the time-average of the poynting vector is `<S> = Re(S)/2 = Re(E x H) / 2`.
        The factor of `1/2` can be omitted if root-mean-square quantities are used
        instead.

    Args:
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`

    Returns:
        Function `f` that returns E x H as required for the poynting vector.
    """
    def exh(e: cfdfield_t, h: cfdfield_t) -> cfdfield_t:
        s = numpy.empty_like(e)
        ex = e[0] * dxes[0][0][:, None, None]
        ey = e[1] * dxes[0][1][None, :, None]
        ez = e[2] * dxes[0][2][None, None, :]
        hx = h[0] * dxes[1][0][:, None, None]
        hy = h[1] * dxes[1][1][None, :, None]
        hz = h[2] * dxes[1][2][None, None, :]
        s[0] = numpy.roll(ey, -1, axis=0) * hz - numpy.roll(ez, -1, axis=0) * hy
        s[1] = numpy.roll(ez, -1, axis=1) * hx - numpy.roll(ex, -1, axis=1) * hz
        s[2] = numpy.roll(ex, -1, axis=2) * hy - numpy.roll(ey, -1, axis=2) * hx
        return s
    return exh
