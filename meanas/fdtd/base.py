"""
Basic FDTD field updates
"""
from typing import List, Callable, Tuple, Dict
import numpy

from ..fdmath import dx_lists_t, fdfield_t, fdfield_updater_t
from ..fdmath.functional import curl_forward, curl_back


__author__ = 'Jan Petykiewicz'


def maxwell_e(dt: float, dxes: dx_lists_t = None) -> fdfield_updater_t:
    if dxes is not None:
        curl_h_fun = curl_back(dxes[1])
    else:
        curl_h_fun = curl_back()

    def me_fun(e: fdfield_t, h: fdfield_t, epsilon: fdfield_t):
        e += dt * curl_h_fun(h) / epsilon
        return e

    return me_fun


def maxwell_h(dt: float, dxes: dx_lists_t = None) -> fdfield_updater_t:
    if dxes is not None:
        curl_e_fun = curl_forward(dxes[0])
    else:
        curl_e_fun = curl_forward()

    def mh_fun(e: fdfield_t, h: fdfield_t):
        h -= dt * curl_e_fun(e)
        return h

    return mh_fun
