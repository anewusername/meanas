"""
Math functions for finite difference simulations

Basic discrete calculus etc.
"""
from typing import List, Callable, Tuple, Dict
import numpy

from .. import field_t, field_updater


def deriv_forward(dx_e: List[numpy.ndarray] = None) -> field_updater:
    """
    Utility operators for taking discretized derivatives (backward variant).

    Args:
        dx_e: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        List of functions for taking forward derivatives along each axis.
    """
    if dx_e:
        derivs = [lambda f: (numpy.roll(f, -1, axis=0) - f) / dx_e[0][:, None, None],
                  lambda f: (numpy.roll(f, -1, axis=1) - f) / dx_e[1][None, :, None],
                  lambda f: (numpy.roll(f, -1, axis=2) - f) / dx_e[2][None, None, :]]
    else:
        derivs = [lambda f: numpy.roll(f, -1, axis=0) - f,
                  lambda f: numpy.roll(f, -1, axis=1) - f,
                  lambda f: numpy.roll(f, -1, axis=2) - f]
    return derivs


def deriv_back(dx_h: List[numpy.ndarray] = None) -> field_updater:
    """
    Utility operators for taking discretized derivatives (forward variant).

    Args:
        dx_h: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        List of functions for taking forward derivatives along each axis.
    """
    if dx_h:
        derivs = [lambda f: (f - numpy.roll(f, 1, axis=0)) / dx_h[0][:, None, None],
                  lambda f: (f - numpy.roll(f, 1, axis=1)) / dx_h[1][None, :, None],
                  lambda f: (f - numpy.roll(f, 1, axis=2)) / dx_h[2][None, None, :]]
    else:
        derivs = [lambda f: f - numpy.roll(f, 1, axis=0),
                  lambda f: f - numpy.roll(f, 1, axis=1),
                  lambda f: f - numpy.roll(f, 1, axis=2)]
    return derivs


def curl_forward(dx_e: List[numpy.ndarray] = None) -> field_updater:
    """
    Curl operator for use with the E field.

    Args:
        dx_e: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        Function `f` for taking the discrete forward curl of a field,
        `f(E)` -> curlE \\( = \\nabla_f \\times E \\)
    """
    Dx, Dy, Dz = deriv_forward(dx_e)

    def ce_fun(e: field_t) -> field_t:
        output = numpy.empty_like(e)
        output[0] = Dy(e[2])
        output[1] = Dz(e[0])
        output[2] = Dx(e[1])
        output[0] -= Dz(e[1])
        output[1] -= Dx(e[2])
        output[2] -= Dy(e[0])
        return output

    return ce_fun


def curl_back(dx_h: List[numpy.ndarray] = None) -> field_updater:
    """
    Create a function which takes the backward curl of a field.

    Args:
        dx_h: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        Function `f` for taking the discrete backward curl of a field,
        `f(H)` -> curlH \\( = \\nabla_b \\times H \\)
    """
    Dx, Dy, Dz = deriv_back(dx_h)

    def ch_fun(h: field_t) -> field_t:
        output = numpy.empty_like(h)
        output[0] = Dy(h[2])
        output[1] = Dz(h[0])
        output[2] = Dx(h[1])
        output[0] -= Dz(h[1])
        output[1] -= Dx(h[2])
        output[2] -= Dy(h[0])
        return output

    return ch_fun

