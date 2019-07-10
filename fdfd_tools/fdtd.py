from typing import List, Callable, Tuple, Dict
import numpy

from . import dx_lists_t, field_t

__author__ = 'Jan Petykiewicz'


functional_matrix = Callable[[field_t], field_t]


def curl_h(dxes: dx_lists_t = None) -> functional_matrix:
    """
    Curl operator for use with the H field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Function for taking the discretized curl of the H-field, F(H) -> curlH
    """
    if dxes:
        dxyz_b = numpy.meshgrid(*dxes[1], indexing='ij')

        def dh(f, ax):
            return (f - numpy.roll(f, 1, axis=ax)) / dxyz_b[ax]
    else:
        def dh(f, ax):
            return f - numpy.roll(f, 1, axis=ax)

    def ch_fun(h: field_t) -> field_t:
        output = numpy.empty_like(h)
        output[0] = dh(h[2], 1)
        output[1] = dh(h[0], 2)
        output[2] = dh(h[1], 0)
        output[0] -= dh(h[1], 2)
        output[1] -= dh(h[2], 0)
        output[2] -= dh(h[0], 1)
        return output

    return ch_fun


def curl_e(dxes: dx_lists_t = None) -> functional_matrix:
    """
    Curl operator for use with the E field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Function for taking the discretized curl of the E-field, F(E) -> curlE
    """
    if dxes is not None:
        dxyz_a = numpy.meshgrid(*dxes[0], indexing='ij')

        def de(f, ax):
            return (numpy.roll(f, -1, axis=ax) - f) / dxyz_a[ax]
    else:
        def de(f, ax):
            return numpy.roll(f, -1, axis=ax) - f

    def ce_fun(e: field_t) -> field_t:
        output = numpy.empty_like(e)
        output[0] = de(e[2], 1)
        output[1] = de(e[0], 2)
        output[2] = de(e[1], 0)
        output[0] -= de(e[1], 2)
        output[1] -= de(e[2], 0)
        output[2] -= de(e[0], 1)
        return output

    return ce_fun


def maxwell_e(dt: float, dxes: dx_lists_t = None) -> functional_matrix:
    curl_h_fun = curl_h(dxes)

    def me_fun(e: field_t, h: field_t, epsilon: field_t):
        ch = curl_h_fun(h)
        for ei, ci, epsi in zip(e, ch, epsilon):
            ei += dt * ci / epsi
        return e

    return me_fun


def maxwell_h(dt: float, dxes: dx_lists_t = None) -> functional_matrix:
    curl_e_fun = curl_e(dxes)

    def mh_fun(e: field_t, h: field_t):
        ce = curl_e_fun(e)
        for hi, ci in zip(h, ce):
            hi -= dt * ci
        return h

    return mh_fun


def conducting_boundary(direction: int,
                        polarity: int
                        ) -> Tuple[functional_matrix, functional_matrix]:
    dirs = [0, 1, 2]
    if direction not in dirs:
        raise Exception('Invalid direction: {}'.format(direction))
    dirs.remove(direction)
    u, v = dirs

    if polarity < 0:
        boundary_slice = [slice(None)] * 3
        shifted1_slice = [slice(None)] * 3
        boundary_slice[direction] = 0
        shifted1_slice[direction] = 1

        def en(e: field_t):
            e[direction][boundary_slice] = 0
            e[u][boundary_slice] = e[u][shifted1_slice]
            e[v][boundary_slice] = e[v][shifted1_slice]
            return e

        def hn(h: field_t):
            h[direction][boundary_slice] = h[direction][shifted1_slice]
            h[u][boundary_slice] = 0
            h[v][boundary_slice] = 0
            return h

        return en, hn

    elif polarity > 0:
        boundary_slice = [slice(None)] * 3
        shifted1_slice = [slice(None)] * 3
        shifted2_slice = [slice(None)] * 3
        boundary_slice[direction] = -1
        shifted1_slice[direction] = -2
        shifted2_slice[direction] = -3

        def ep(e: field_t):
            e[direction][boundary_slice] = -e[direction][shifted2_slice]
            e[direction][shifted1_slice] = 0
            e[u][boundary_slice] = e[u][shifted1_slice]
            e[v][boundary_slice] = e[v][shifted1_slice]
            return e

        def hp(h: field_t):
            h[direction][boundary_slice] = h[direction][shifted1_slice]
            h[u][boundary_slice] = -h[u][shifted2_slice]
            h[u][shifted1_slice] = 0
            h[v][boundary_slice] = -h[v][shifted2_slice]
            h[v][shifted1_slice] = 0
            return h

        return ep, hp

    else:
        raise Exception('Bad polarity: {}'.format(polarity))


def cpml(direction:int,
         polarity: int,
         dt: float,
         epsilon: field_t,
         thickness: int = 8,
         epsilon_eff: float = 1,
         dtype: numpy.dtype = numpy.float32,
         ) -> Tuple[Callable, Callable, Dict[str, field_t]]:

    if direction not in range(3):
        raise Exception('Invalid direction: {}'.format(direction))

    if polarity not in (-1, 1):
        raise Exception('Invalid polarity: {}'.format(polarity))

    if thickness <= 2:
        raise Exception('It would be wise to have a pml with 4+ cells of thickness')

    if epsilon_eff <= 0:
        raise Exception('epsilon_eff must be positive')

    m = (3.5, 1)
    sigma_max = 0.8 * (m[0] + 1) / numpy.sqrt(epsilon_eff)
    alpha_max = 0  # TODO: Decide what to do about non-zero alpha
    transverse = numpy.delete(range(3), direction)
    u, v = transverse

    xe = numpy.arange(1, thickness+1, dtype=float)
    xh = numpy.arange(1, thickness+1, dtype=float)
    if polarity > 0:
        xe -= 0.5
    elif polarity < 0:
        xh -= 0.5
        xe = xe[::-1]
        xh = xh[::-1]
    else:
        raise Exception('Bad polarity!')

    expand_slice = [None] * 3
    expand_slice[direction] = slice(None)

    def par(x):
        sigma = ((x / thickness) ** m[0]) * sigma_max
        alpha = ((1 - x / thickness) ** m[1]) * alpha_max
        p0 = numpy.exp(-(sigma + alpha) * dt)
        p1 = sigma / (sigma + alpha) * (p0 - 1)
        return p0[expand_slice], p1[expand_slice]

    p0e, p1e = par(xe)
    p0h, p1h = par(xh)

    region = [slice(None)] * 3
    if polarity < 0:
        region[direction] = slice(None, thickness)
    elif polarity > 0:
        region[direction] = slice(-thickness, None)
    else:
        raise Exception('Bad polarity!')

    if direction == 1:
        se = 1
    else:
        se = -1

    # TODO check if epsilon is uniform?
    shape = list(epsilon[0].shape)
    shape[direction] = thickness
    psi_e = [numpy.zeros(shape, dtype=dtype), numpy.zeros(shape, dtype=dtype)]
    psi_h = [numpy.zeros(shape, dtype=dtype), numpy.zeros(shape, dtype=dtype)]

    fields = {
        'psi_e_u': psi_e[0],
        'psi_e_v': psi_e[1],
        'psi_h_u': psi_h[0],
        'psi_h_v': psi_h[1],
    }

    def pml_e(e: field_t, h: field_t, epsilon: field_t) -> Tuple[field_t, field_t]:
        psi_e[0] *= p0e
        psi_e[0] += p1e * (h[v][region] - numpy.roll(h[v], 1, axis=direction)[region])
        psi_e[1] *= p0e
        psi_e[1] += p1e * (h[u][region] - numpy.roll(h[u], 1, axis=direction)[region])
        e[u][region] += se * dt * psi_e[0] / epsilon[u][region]
        e[v][region] -= se * dt * psi_e[1] / epsilon[v][region]
        return e, h

    def pml_h(e: field_t, h: field_t) -> Tuple[field_t, field_t]:
        psi_h[0] *= p0h
        psi_h[0] += p1h * (numpy.roll(e[v], -1, axis=direction)[region] - e[v][region])
        psi_h[1] *= p0h
        psi_h[1] += p1h * (numpy.roll(e[u], -1, axis=direction)[region] - e[u][region])
        h[u][region] -= se * dt * psi_h[0]
        h[v][region] += se * dt * psi_h[1]
        return e, h

    return pml_e, pml_h, fields
