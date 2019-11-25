"""
Basic FDTD field updates
"""
from typing import List, Callable, Tuple, Dict
import numpy

from .. import dx_lists_t, field_t, field_updater
from ..fdmath.functional import curl_forward, curl_back


__author__ = 'Jan Petykiewicz'


def maxwell_e(dt: float, dxes: dx_lists_t = None) -> field_updater:
    curl_h_fun = curl_back(dxes[1])

    def me_fun(e: field_t, h: field_t, epsilon: field_t):
        e += dt * curl_h_fun(h) / epsilon
        return e

    return me_fun


def maxwell_h(dt: float, dxes: dx_lists_t = None) -> field_updater:
    curl_e_fun = curl_forward(dxes[0])

    def mh_fun(e: field_t, h: field_t):
        h -= dt * curl_e_fun(e)
        return h

    return mh_fun
