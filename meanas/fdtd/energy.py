# pylint: disable=unsupported-assignment-operation
from typing import List, Callable, Tuple, Dict
import numpy

from .. import dx_lists_t, field_t, field_updater


def poynting(e: field_t,
             h: field_t,
             dxes: dx_lists_t = None,
             ) -> field_t:
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    ex = e[0] * dxes[0][0][:, None, None]
    ey = e[1] * dxes[0][1][None, :, None]
    ez = e[2] * dxes[0][2][None, None, :]
    hx = h[0] * dxes[1][0][:, None, None]
    hy = h[1] * dxes[1][1][None, :, None]
    hz = h[2] * dxes[1][2][None, None, :]

    s = numpy.empty_like(e)
    s[0] = numpy.roll(ey, -1, axis=0) * hz - numpy.roll(ez, -1, axis=0) * hy
    s[1] = numpy.roll(ez, -1, axis=1) * hx - numpy.roll(ex, -1, axis=1) * hz
    s[2] = numpy.roll(ex, -1, axis=2) * hy - numpy.roll(ey, -1, axis=2) * hx
    return s


def poynting_divergence(s: field_t = None,
                        *,
                        e: field_t = None,
                        h: field_t = None,
                        dxes: dx_lists_t = None,
                        ) -> field_t:
    if s is None:
        s = poynting(e, h, dxes=dxes)

    #TODO use deriv operators
    ds = ((s[0] - numpy.roll(s[0], 1, axis=0)) +
          (s[1] - numpy.roll(s[1], 1, axis=1)) +
          (s[2] - numpy.roll(s[2], 1, axis=2)))
    return ds


def energy_hstep(e0: field_t,
                 h1: field_t,
                 e2: field_t,
                 epsilon: field_t = None,
                 mu: field_t = None,
                 dxes: dx_lists_t = None,
                 ) -> field_t:
    u = dxmul(e0 * e2, h1 * h1, epsilon, mu, dxes)
    return u


def energy_estep(h0: field_t,
                 e1: field_t,
                 h2: field_t,
                 epsilon: field_t = None,
                 mu: field_t = None,
                 dxes: dx_lists_t = None,
                 ) -> field_t:
    u = dxmul(e1 * e1, h0 * h2, epsilon, mu, dxes)
    return u


def delta_energy_h2e(dt: float,
                     e0: field_t,
                     h1: field_t,
                     e2: field_t,
                     h3: field_t,
                     epsilon: field_t = None,
                     mu: field_t = None,
                     dxes: dx_lists_t = None,
                     ) -> field_t:
    """
    This is just from (e2 * e2 + h3 * h1) - (h1 * h1 + e0 * e2)
    """
    de = e2 * (e2 - e0) / dt
    dh = h1 * (h3 - h1) / dt
    du = dxmul(de, dh, epsilon, mu, dxes)
    return du


def delta_energy_e2h(dt: float,
                     h0: field_t,
                     e1: field_t,
                     h2: field_t,
                     e3: field_t,
                     epsilon: field_t = None,
                     mu: field_t = None,
                     dxes: dx_lists_t = None,
                     ) -> field_t:
    """
    This is just from (h2 * h2 + e3 * e1) - (e1 * e1 + h0 * h2)
    """
    de = e1 * (e3 - e1) / dt
    dh = h2 * (h2 - h0) / dt
    du = dxmul(de, dh, epsilon, mu, dxes)
    return du


def delta_energy_j(j0: field_t, e1: field_t, dxes: dx_lists_t = None) -> field_t:
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    du = ((j0 * e1).sum(axis=0) *
          dxes[0][0][:, None, None] *
          dxes[0][1][None, :, None] *
          dxes[0][2][None, None, :])
    return du


def dxmul(ee: field_t,
          hh: field_t,
          epsilon: field_t = None,
          mu: field_t = None,
          dxes: dx_lists_t = None
          ) -> field_t:
    if epsilon is None:
        epsilon = 1
    if mu is None:
        mu = 1
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    result = ((ee * epsilon).sum(axis=0) *
              dxes[0][0][:, None, None] *
              dxes[0][1][None, :, None] *
              dxes[0][2][None, None, :] +
              (hh * mu).sum(axis=0) *
              dxes[1][0][:, None, None] *
              dxes[1][1][None, :, None] *
              dxes[1][2][None, None, :])
    return result
