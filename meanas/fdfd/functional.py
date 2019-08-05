"""
Functional versions of many FDFD operators. These can be useful for performing
 FDFD calculations without needing to construct large matrices in memory.

The functions generated here expect field inputs with shape (3, X, Y, Z),
e.g. E = [E_x, E_y, E_z] where each component has shape (X, Y, Z)
"""
from typing import List, Callable
import numpy

from .. import dx_lists_t, field_t

__author__ = 'Jan Petykiewicz'


functional_matrix = Callable[[field_t], field_t]


def curl_h(dxes: dx_lists_t) -> functional_matrix:
    """
    Curl operator for use with the H field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :return: Function for taking the discretized curl of the H-field, F(H) -> curlH
    """
    dxyz_b = numpy.meshgrid(*dxes[1], indexing='ij')

    def dh(f, ax):
        return (f - numpy.roll(f, 1, axis=ax)) / dxyz_b[ax]

    def ch_fun(h: field_t) -> field_t:
        e = [dh(h[2], 1) - dh(h[1], 2),
             dh(h[0], 2) - dh(h[2], 0),
             dh(h[1], 0) - dh(h[0], 1)]
        return e

    return ch_fun


def curl_e(dxes: dx_lists_t) -> functional_matrix:
    """
    Curl operator for use with the E field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :return: Function for taking the discretized curl of the E-field, F(E) -> curlE
    """
    dxyz_a = numpy.meshgrid(*dxes[0], indexing='ij')

    def de(f, ax):
        return (numpy.roll(f, -1, axis=ax) - f) / dxyz_a[ax]

    def ce_fun(e: field_t) -> field_t:
        h = [de(e[2], 1) - de(e[1], 2),
             de(e[0], 2) - de(e[2], 0),
             de(e[1], 0) - de(e[0], 1)]
        return h

    return ce_fun


def e_full(omega: complex,
           dxes: dx_lists_t,
           epsilon: field_t,
           mu: field_t = None
           ) -> functional_matrix:
    """
    Wave operator del x (1/mu * del x) - omega**2 * epsilon, for use with E-field,
     with wave equation
    (del x (1/mu * del x) - omega**2 * epsilon) E = -i * omega * J

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: Function implementing the wave operator A(E) -> E
    """
    ch = curl_h(dxes)
    ce = curl_e(dxes)

    def op_1(e):
        curls = ch(ce(e))
        return [c - omega ** 2 * e * x for c, e, x in zip(curls, epsilon, e)]

    def op_mu(e):
        curls = ch([m * y for m, y in zip(mu, ce(e))])
        return [c - omega ** 2 * p * x for c, p, x in zip(curls, epsilon, e)]

    if numpy.any(numpy.equal(mu, None)):
        return op_1
    else:
        return op_mu


def eh_full(omega: complex,
            dxes: dx_lists_t,
            epsilon: field_t,
            mu: field_t = None
            ) -> functional_matrix:
    """
    Wave operator for full (both E and H) field representation.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: Function implementing the wave operator A(E, H) -> (E, H)
    """
    ch = curl_h(dxes)
    ce = curl_e(dxes)

    def op_1(e, h):
        return ([c - 1j * omega * p * x for c, p, x in zip(ch(h), epsilon, e)],
                [c + 1j * omega * y for c, y in zip(ce(e), h)])

    def op_mu(e, h):
        return ([c - 1j * omega * p * x for c, p, x in zip(ch(h), epsilon, e)],
                [c + 1j * omega * m * y for c, m, y in zip(ce(e), mu, h)])

    if numpy.any(numpy.equal(mu, None)):
        return op_1
    else:
        return op_mu


def e2h(omega: complex,
        dxes: dx_lists_t,
        mu: field_t = None,
        ) -> functional_matrix:
    """
   Utility operator for converting the E field into the H field.
   For use with e_full -- assumes that there is no magnetic current M.

   :param omega: Angular frequency of the simulation
   :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
   :param mu: Magnetic permeability (default 1 everywhere)
   :return: Function for converting E to H
   """
    A2 = curl_e(dxes)

    def e2h_1_1(e):
        return [y / (-1j * omega) for y in A2(e)]

    def e2h_mu(e):
        return [y / (-1j * omega * m) for y, m in zip(A2(e), mu)]

    if numpy.any(numpy.equal(mu, None)):
        return e2h_1_1
    else:
        return e2h_mu


def m2j(omega: complex,
        dxes: dx_lists_t,
        mu: field_t = None,
        ) -> functional_matrix:
    """
   Utility operator for converting magnetic current (M) distribution
   into equivalent electric current distribution (J).
   For use with e.g. e_full().

   :param omega: Angular frequency of the simulation
   :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
   :param mu: Magnetic permeability (default 1 everywhere)
   :return: Function for converting M to J
   """
    ch = curl_h(dxes)

    def m2j_mu(m):
        m_mu = [m[k] / mu[k] for k in range[3]]
        J = [Ji / (-1j * omega) for Ji in ch(m_mu)]
        return J

    def m2j_1(m):
        J = [Ji / (-1j * omega) for Ji in ch(m)]
        return J

    if numpy.any(numpy.equal(mu, None)):
        return m2j_1
    else:
        return m2j_mu


