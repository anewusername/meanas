"""
Functional versions of many FDFD operators. These can be useful for performing
 FDFD calculations without needing to construct large matrices in memory.

The functions generated here expect inputs in the form E = [E_x, E_y, E_z], where each
 component E_* is an ndarray of equal shape.
"""
from typing import List, Callable
import numpy

from . import dx_lists_t, field_t

__author__ = 'Jan Petykiewicz'


functional_matrix = Callable[[List[numpy.ndarray]], List[numpy.ndarray]]


def curl_h(dxes: dx_lists_t) -> functional_matrix:
    """
    Curl operator for use with the H field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Function for taking the discretized curl of the H-field, F(H) -> curlH
    """
    dxyz_b = numpy.meshgrid(*dxes[1], indexing='ij')

    def dH(f, ax):
        return (f - numpy.roll(f, 1, axis=ax)) / dxyz_b[ax]

    def ch_fun(H: List[numpy.ndarray]) -> List[numpy.ndarray]:
        E = [dH(H[2], 1) - dH(H[1], 2),
             dH(H[0], 2) - dH(H[2], 0),
             dH(H[1], 0) - dH(H[0], 1)]
        return E

    return ch_fun


def curl_e(dxes: dx_lists_t) -> functional_matrix:
    """
    Curl operator for use with the E field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Function for taking the discretized curl of the E-field, F(E) -> curlE
    """
    dxyz_a = numpy.meshgrid(*dxes[0], indexing='ij')

    def dE(f, ax):
        return (numpy.roll(f, -1, axis=ax) - f) / dxyz_a[ax]

    def ce_fun(E: List[numpy.ndarray]) -> List[numpy.ndarray]:
        H = [dE(E[2], 1) - dE(E[1], 2),
             dE(E[0], 2) - dE(E[2], 0),
             dE(E[1], 0) - dE(E[0], 1)]
        return H

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
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: Function implementing the wave operator A(E) -> E
    """
    ch = curl_h(dxes)
    ce = curl_e(dxes)

    def op_1(E):
        curls = ch(ce(E))
        return [c - omega ** 2 * e * x for c, e, x in zip(curls, epsilon, E)]

    def op_mu(E):
        curls = ch([m * y for m, y in zip(mu, ce(E))])
        return [c - omega ** 2 * e * x for c, e, x in zip(curls, epsilon, E)]

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
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: Function implementing the wave operator A(E, H) -> (E, H)
    """
    ch = curl_h(dxes)
    ce = curl_e(dxes)

    def op_1(E, H):
        return ([c - 1j * omega * e * x for c, e, x in zip(ch(H), epsilon, E)],
                [c + 1j * omega * y for c, y in zip(ce(E), H)])

    def op_mu(E, H):
        return ([c - 1j * omega * e * x for c, e, x in zip(ch(H), epsilon, E)],
                [c + 1j * omega * m * y for c, m, y in zip(ce(E), mu, H)])

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
   :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
   :param mu: Magnetic permeability (default 1 everywhere)
   :return: Function for converting E to H
   """
    A2 = curl_e(dxes)

    def e2h_1_1(E):
        return [y / (-1j * omega) for y in A2(E)]

    def e2h_mu(E):
        return [y / (-1j * omega * m) for y, m in zip(A2(E), mu)]

    if numpy.any(numpy.equal(mu, None)):
        return e2h_1_1
    else:
        return e2h_mu
