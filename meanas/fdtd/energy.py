from typing import List, Callable, Tuple, Dict
import numpy

from .. import dx_lists_t, field_t, field_updater


def poynting(e, h):
    s = (numpy.roll(e[1], -1, axis=0) * h[2] - numpy.roll(e[2], -1, axis=0) * h[1],
         numpy.roll(e[2], -1, axis=1) * h[0] - numpy.roll(e[0], -1, axis=1) * h[2],
         numpy.roll(e[0], -1, axis=2) * h[1] - numpy.roll(e[1], -1, axis=2) * h[0])
    return numpy.array(s)


def poynting_divergence(s=None, *, e=None, h=None, dxes=None): # TODO dxes
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    if s is None:
        s = poynting(e, h)

    ds = ((s[0] - numpy.roll(s[0], 1, axis=0)) / numpy.sqrt(dxes[0][0] * dxes[1][0])[:, None, None] +
          (s[1] - numpy.roll(s[1], 1, axis=1)) / numpy.sqrt(dxes[0][1] * dxes[1][1])[None, :, None] +
          (s[2] - numpy.roll(s[2], 1, axis=2)) / numpy.sqrt(dxes[0][2] * dxes[1][2])[None, None, :] )
    return ds


def energy_hstep(e0, h1, e2, epsilon=None, mu=None, dxes=None):
    u = dxmul(e0 * e2, h1 * h1, epsilon, mu, dxes)
    return u


def energy_estep(h0, e1, h2, epsilon=None, mu=None, dxes=None):
    u = dxmul(e1 * e1, h0 * h2, epsilon, mu, dxes)
    return u


def delta_energy_h2e(dt, e0, h1, e2, h3, epsilon=None, mu=None, dxes=None):
    """
    This is just from (e2 * e2 + h3 * h1) - (h1 * h1 + e0 * e2)
    """
    de = e2 * (e2 - e0) / dt
    dh = h1 * (h3 - h1) / dt
    du = dxmul(de, dh, epsilon, mu, dxes)
    return du


def delta_energy_e2h(dt, h0, e1, h2, e3, epsilon=None, mu=None, dxes=None):
    """
    This is just from (h2 * h2 + e3 * e1) - (e1 * e1 + h0 * h2)
    """
    de = e1 * (e3 - e1) / dt
    dh = h2 * (h2 - h0) / dt
    du = dxmul(de, dh, epsilon, mu, dxes)
    return du


def delta_energy_j(j0, e1, dxes=None):
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    du = ((j0 * e1).sum(axis=0) *
          dxes[0][0][:, None, None] *
          dxes[0][1][None, :, None] *
          dxes[0][2][None, None, :])
    return du


def dxmul(ee, hh, epsilon=None, mu=None, dxes=None):
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
