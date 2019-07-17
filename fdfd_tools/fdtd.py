from typing import List, Callable, Tuple, Dict
import numpy

from . import dx_lists_t, field_t

#TODO fix pmls

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
        e += dt * curl_h_fun(h) / epsilon
        return e

    return me_fun


def maxwell_h(dt: float, dxes: dx_lists_t = None) -> functional_matrix:
    curl_e_fun = curl_e(dxes)

    def mh_fun(e: field_t, h: field_t):
        h -= dt * curl_e_fun(e)
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
         ln_R_per_layer: float = -1.6,
         epsilon_eff: float = 1,
         mu_eff: float = 1,
         m: float = 3.5,
         ma: float = 1,
         cfs_alpha: float = 0,
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

    sigma_max = -ln_R_per_layer / 2 * (m + 1)
    kappa_max = numpy.sqrt(epsilon_eff * mu_eff)
    alpha_max = cfs_alpha
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
        scaling = (x / thickness) ** m
        sigma = scaling * sigma_max
        kappa = 1 + scaling * (kappa_max - 1)
        alpha = ((1 - x / thickness) ** ma) * alpha_max
        p0 = numpy.exp(-(sigma / kappa + alpha) * dt)
        p1 = sigma / (sigma + kappa * alpha) * (p0 - 1)
        p2 = 1 / kappa
        return p0[expand_slice], p1[expand_slice], p2[expand_slice]

    p0e, p1e, p2e = par(xe)
    p0h, p1h, p2h = par(xh)

    region = [slice(None)] * 3
    if polarity < 0:
        region[direction] = slice(None, thickness)
    elif polarity > 0:
        region[direction] = slice(-thickness, None)
    else:
        raise Exception('Bad polarity!')

    se = 1 if direction == 1 else -1

    # TODO check if epsilon is uniform in pml region?
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

    # Note that this is kinda slow -- would be faster to reuse dHv*p2h for the original
    #  H update, but then you have multiple arrays and a monolithic (field + pml) update operation
    def pml_e(e: field_t, h: field_t, epsilon: field_t) -> Tuple[field_t, field_t]:
        dHv = h[v][region] - numpy.roll(h[v], 1, axis=direction)[region]
        dHu = h[u][region] - numpy.roll(h[u], 1, axis=direction)[region]
        psi_e[0] *= p0e
        psi_e[0] += p1e * dHv * p2e
        psi_e[1] *= p0e
        psi_e[1] += p1e * dHu * p2e
        e[u][region] += se * dt / epsilon[u][region] * (psi_e[0] + (p2e - 1) * dHv)
        e[v][region] -= se * dt / epsilon[v][region] * (psi_e[1] + (p2e - 1) * dHu)
        return e, h

    def pml_h(e: field_t, h: field_t) -> Tuple[field_t, field_t]:
        dEv = (numpy.roll(e[v], -1, axis=direction)[region] - e[v][region])
        dEu = (numpy.roll(e[u], -1, axis=direction)[region] - e[u][region])
        psi_h[0] *= p0h
        psi_h[0] += p1h * dEv * p2h
        psi_h[1] *= p0h
        psi_h[1] += p1h * dEu * p2h
        h[u][region] -= se * dt * (psi_h[0] + (p2h - 1) * dEv)
        h[v][region] += se * dt * (psi_h[1] + (p2h - 1) * dEu)
        return e, h

    return pml_e, pml_h, fields


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
    du = dt * dxmul(de, dh, epsilon, mu, dxes)
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



