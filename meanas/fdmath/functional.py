"""
Math functions for finite difference simulations

Basic discrete calculus etc.
"""
from typing import Sequence, Tuple, Optional, Callable

import numpy
from numpy.typing import NDArray

from .types import fdfield_t, fdfield_updater_t


def deriv_forward(
        dx_e: Optional[Sequence[NDArray[numpy.float_]]] = None,
        ) -> Tuple[fdfield_updater_t, fdfield_updater_t, fdfield_updater_t]:
    """
    Utility operators for taking discretized derivatives (backward variant).

    Args:
        dx_e: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        List of functions for taking forward derivatives along each axis.
    """
    if dx_e is not None:
        derivs = (lambda f: (numpy.roll(f, -1, axis=0) - f) / dx_e[0][:, None, None],
                  lambda f: (numpy.roll(f, -1, axis=1) - f) / dx_e[1][None, :, None],
                  lambda f: (numpy.roll(f, -1, axis=2) - f) / dx_e[2][None, None, :])
    else:
        derivs = (lambda f: numpy.roll(f, -1, axis=0) - f,
                  lambda f: numpy.roll(f, -1, axis=1) - f,
                  lambda f: numpy.roll(f, -1, axis=2) - f)
    return derivs


def deriv_back(
        dx_h: Optional[Sequence[NDArray[numpy.float_]]] = None,
        ) -> Tuple[fdfield_updater_t, fdfield_updater_t, fdfield_updater_t]:
    """
    Utility operators for taking discretized derivatives (forward variant).

    Args:
        dx_h: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        List of functions for taking forward derivatives along each axis.
    """
    if dx_h is not None:
        derivs = (lambda f: (f - numpy.roll(f, 1, axis=0)) / dx_h[0][:, None, None],
                  lambda f: (f - numpy.roll(f, 1, axis=1)) / dx_h[1][None, :, None],
                  lambda f: (f - numpy.roll(f, 1, axis=2)) / dx_h[2][None, None, :])
    else:
        derivs = (lambda f: f - numpy.roll(f, 1, axis=0),
                  lambda f: f - numpy.roll(f, 1, axis=1),
                  lambda f: f - numpy.roll(f, 1, axis=2))
    return derivs


def curl_forward(
        dx_e: Optional[Sequence[NDArray[numpy.float_]]] = None,
        ) -> fdfield_updater_t:
    """
    Curl operator for use with the E field.

    Args:
        dx_e: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        Function `f` for taking the discrete forward curl of a field,
        `f(E)` -> curlE $= \\nabla_f \\times E$
    """
    Dx, Dy, Dz = deriv_forward(dx_e)

    def ce_fun(e: fdfield_t) -> fdfield_t:
        output = numpy.empty_like(e)
        output[0] = Dy(e[2])
        output[1] = Dz(e[0])
        output[2] = Dx(e[1])
        output[0] -= Dz(e[1])
        output[1] -= Dx(e[2])
        output[2] -= Dy(e[0])
        return output

    return ce_fun


def curl_back(
        dx_h: Optional[Sequence[NDArray[numpy.float_]]] = None,
        ) -> fdfield_updater_t:
    """
    Create a function which takes the backward curl of a field.

    Args:
        dx_h: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        Function `f` for taking the discrete backward curl of a field,
        `f(H)` -> curlH $= \\nabla_b \\times H$
    """
    Dx, Dy, Dz = deriv_back(dx_h)

    def ch_fun(h: fdfield_t) -> fdfield_t:
        output = numpy.empty_like(h)
        output[0] = Dy(h[2])
        output[1] = Dz(h[0])
        output[2] = Dx(h[1])
        output[0] -= Dz(h[1])
        output[1] -= Dx(h[2])
        output[2] -= Dy(h[0])
        return output

    return ch_fun


def curl_forward_parts(
        dx_e: Optional[Sequence[NDArray[numpy.float_]]] = None,
        ) -> Callable:
    Dx, Dy, Dz = deriv_forward(dx_e)

    def mkparts_fwd(e: fdfield_t) -> Tuple[Tuple[fdfield_t, fdfield_t], ...]:
        return ((-Dz(e[1]),  Dy(e[2])),
                ( Dz(e[0]), -Dx(e[2])),
                (-Dy(e[0]),  Dx(e[1])))

    return mkparts_fwd


def curl_back_parts(
        dx_h: Optional[Sequence[NDArray[numpy.float_]]] = None,
        ) -> Callable:
    Dx, Dy, Dz = deriv_back(dx_h)

    def mkparts_back(h: fdfield_t) -> Tuple[Tuple[fdfield_t, fdfield_t], ...]:
        return ((-Dz(h[1]),  Dy(h[2])),
                ( Dz(h[0]), -Dx(h[2])),
                (-Dy(h[0]),  Dx(h[1])))

    return mkparts_back
