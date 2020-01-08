"""
PML implementations

#TODO discussion of PMLs
#TODO cpml documentation

"""
# TODO retest pmls!

from typing import List, Callable, Tuple, Dict
import numpy

from ..fdmath import dx_lists_t, fdfield_t, fdfield_updater_t


__author__ = 'Jan Petykiewicz'


def cpml(direction: int,
         polarity: int,
         dt: float,
         epsilon: fdfield_t,
         thickness: int = 8,
         ln_R_per_layer: float = -1.6,
         epsilon_eff: float = 1,
         mu_eff: float = 1,
         m: float = 3.5,
         ma: float = 1,
         cfs_alpha: float = 0,
         dtype: numpy.dtype = numpy.float32,
         ) -> Tuple[Callable, Callable, Dict[str, fdfield_t]]:

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
    expand_slice = tuple(expand_slice)

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
    region = tuple(region)

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
    def pml_e(e: fdfield_t, h: fdfield_t, epsilon: fdfield_t) -> Tuple[fdfield_t, fdfield_t]:
        dHv = h[v][region] - numpy.roll(h[v], 1, axis=direction)[region]
        dHu = h[u][region] - numpy.roll(h[u], 1, axis=direction)[region]
        psi_e[0] *= p0e
        psi_e[0] += p1e * dHv * p2e
        psi_e[1] *= p0e
        psi_e[1] += p1e * dHu * p2e
        e[u][region] += se * dt / epsilon[u][region] * (psi_e[0] + (p2e - 1) * dHv)
        e[v][region] -= se * dt / epsilon[v][region] * (psi_e[1] + (p2e - 1) * dHu)
        return e, h

    def pml_h(e: fdfield_t, h: fdfield_t) -> Tuple[fdfield_t, fdfield_t]:
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
