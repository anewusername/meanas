"""
Basic FDTD field updates
"""
from typing import List, Callable, Tuple, Dict
import numpy

from .. import dx_lists_t, field_t, field_updater

__author__ = 'Jan Petykiewicz'


def curl_h(dxes: dx_lists_t = None) -> field_updater:
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


def curl_e(dxes: dx_lists_t = None) -> field_updater:
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


def maxwell_e(dt: float, dxes: dx_lists_t = None) -> field_updater:
    curl_h_fun = curl_h(dxes)

    def me_fun(e: field_t, h: field_t, epsilon: field_t):
        e += dt * curl_h_fun(h) / epsilon
        return e

    return me_fun


def maxwell_h(dt: float, dxes: dx_lists_t = None) -> field_updater:
    curl_e_fun = curl_e(dxes)

    def mh_fun(e: field_t, h: field_t):
        h -= dt * curl_e_fun(e)
        return h

    return mh_fun
