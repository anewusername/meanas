"""
PML implementations

#TODO discussion of PMLs
#TODO cpml documentation

"""
# TODO retest pmls!

from typing import List, Callable, Tuple, Dict, Sequence, Any, Optional
from copy import deepcopy
import numpy
from numpy.typing import NDArray, DTypeLike

from ..fdmath import fdfield_t, dx_lists_t
from ..fdmath.functional import deriv_forward, deriv_back


__author__ = 'Jan Petykiewicz'


def cpml_params(
        axis: int,
        polarity: int,
        dt: float,
        thickness: int = 8,
        ln_R_per_layer: float = -1.6,
        epsilon_eff: float = 1,
        mu_eff: float = 1,
        m: float = 3.5,
        ma: float = 1,
        cfs_alpha: float = 0,
        ) -> Dict[str, Any]:

    if axis not in range(3):
        raise Exception('Invalid axis: {}'.format(axis))

    if polarity not in (-1, 1):
        raise Exception('Invalid polarity: {}'.format(polarity))

    if thickness <= 2:
        raise Exception('It would be wise to have a pml with 4+ cells of thickness')

    if epsilon_eff <= 0:
        raise Exception('epsilon_eff must be positive')

    sigma_max = -ln_R_per_layer / 2 * (m + 1)
    kappa_max = numpy.sqrt(epsilon_eff * mu_eff)
    alpha_max = cfs_alpha

    xe = numpy.arange(1, thickness + 1, dtype=float)        # TODO: pass in dtype?
    xh = numpy.arange(1, thickness + 1, dtype=float)
    if polarity > 0:
        xe -= 0.5
    elif polarity < 0:
        xh -= 0.5
        xe = xe[::-1]
        xh = xh[::-1]
    else:
        raise Exception('Bad polarity!')

    expand_slice_l: List[Any] = [None, None, None]
    expand_slice_l[axis] = slice(None)
    expand_slice = tuple(expand_slice_l)

    def par(x: NDArray[numpy.float64]) -> Tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
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

    region_list = [slice(None), slice(None), slice(None)]
    if polarity < 0:
        region_list[axis] = slice(None, thickness)
    elif polarity > 0:
        region_list[axis] = slice(-thickness, None)
    else:
        raise Exception('Bad polarity!')
    region = tuple(region_list)

    return {
        'param_e': (p0e, p1e, p2e),
        'param_h': (p0h, p1h, p2h),
        'region': region,
        }


def updates_with_cpml(
         cpml_params: Sequence[Sequence[Optional[Dict[str, Any]]]],
         dt: float,
         dxes: dx_lists_t,
         epsilon: fdfield_t,
         *,
         dtype: DTypeLike = numpy.float32,
         ) -> Tuple[Callable[[fdfield_t, fdfield_t, fdfield_t], None],
                    Callable[[fdfield_t, fdfield_t, fdfield_t], None]]:

    Dfx, Dfy, Dfz = deriv_forward(dxes[1])
    Dbx, Dby, Dbz = deriv_back(dxes[1])


    psi_E: List[List[Tuple[Any, Any]]] = [[(None, None) for _ in range(2)] for _ in range(3)]
    psi_H: List[List[Tuple[Any, Any]]] = deepcopy(psi_E)
    params_E: List[List[Tuple[Any, Any, Any, Any]]] = [[(None, None, None, None) for _ in range(2)] for _ in range(3)]
    params_H: List[List[Tuple[Any, Any, Any, Any]]] = deepcopy(params_E)

    for axis in range(3):
        for pp, polarity in enumerate((-1, 1)):
            cpml_param = cpml_params[axis][pp]
            if cpml_param is None:
                psi_E[axis][pp] = (None, None)
                psi_H[axis][pp] = (None, None)
                continue

            region = cpml_param['region']
            region_shape = epsilon[0][region].shape

            psi_E[axis][pp] = (
                numpy.zeros(region_shape, dtype=dtype),
                numpy.zeros(region_shape, dtype=dtype),
                )
            psi_H[axis][pp] = (
                numpy.zeros(region_shape, dtype=dtype),
                numpy.zeros(region_shape, dtype=dtype),
                )
            params_E[axis][pp] = cpml_param['param_e'] + (region,)
            params_H[axis][pp] = cpml_param['param_h'] + (region,)


    pE = numpy.empty_like(epsilon, dtype=dtype)
    pH = numpy.empty_like(epsilon, dtype=dtype)

    def update_E(
            e: fdfield_t,
            h: fdfield_t,
            epsilon: fdfield_t,
            ) -> None:
        dyHx = Dby(h[0])
        dzHx = Dbz(h[0])
        dxHy = Dbx(h[1])
        dzHy = Dbz(h[1])
        dxHz = Dbx(h[2])
        dyHz = Dby(h[2])

        dH = ((dxHy, dxHz),
              (dyHx, dyHz),
              (dzHx, dzHy))

        pE.fill(0)

        for axis in range(3):
            se = (-1, 1, -1)[axis]
            transverse = numpy.delete(range(3), axis)
            u, v = transverse
            dHu, dHv = dH[axis]

            for pp in range(2):
                psi_Eu, psi_Ev = psi_E[axis][pp]

                if psi_Eu is None:
                    # No pml in this direction
                    continue

                p0e, p1e, p2e, region = params_E[axis][pp]

                dHu[region] *= p2e
                dHv[region] *= p2e
                psi_Eu *= p0e
                psi_Ev *= p0e
                psi_Eu += p1e * dHv[region]    # note reversed u,v mapping
                psi_Ev += p1e * dHu[region]
                pE[u][region] += +se * psi_Eu
                pE[v][region] += -se * psi_Ev

        e[0] += dt / epsilon[0] * (dyHz - dzHy + pE[0])
        e[1] += dt / epsilon[1] * (dzHx - dxHz + pE[1])
        e[2] += dt / epsilon[2] * (dxHy - dyHx + pE[2])


    def update_H(
            e: fdfield_t,
            h: fdfield_t,
            mu: fdfield_t = numpy.ones(3),
            ) -> None:
        dyEx = Dfy(e[0])
        dzEx = Dfz(e[0])
        dxEy = Dfx(e[1])
        dzEy = Dfz(e[1])
        dxEz = Dfx(e[2])
        dyEz = Dfy(e[2])

        dE = ((dxEy, dxEz),
              (dyEx, dyEz),
              (dzEx, dzEy))

        pH.fill(0)

        for axis in range(3):
            se = (-1, 1, -1)[axis]
            transverse = numpy.delete(range(3), axis)
            u, v = transverse
            dEu, dEv = dE[axis]

            for pp in range(2):
                psi_Hu, psi_Hv = psi_H[axis][pp]

                if psi_Hu is None:
                    # No pml here
                    continue

                p0h, p1h, p2h, region = params_H[axis][pp]

                dEu[region] *= p2h      # modifies d_E_
                dEv[region] *= p2h
                psi_Hu *= p0h
                psi_Hv *= p0h
                psi_Hu += p1h * dEv[region]    # note reversed u,v mapping
                psi_Hv += p1h * dEu[region]
                pH[u][region] += +se * psi_Hu
                pH[v][region] += -se * psi_Hv

        h[0] -= dt / mu[0] * (dyEz - dzEy + pH[0])
        h[1] -= dt / mu[1] * (dzEx - dxEz + pH[1])
        h[2] -= dt / mu[2] * (dxEy - dyEx + pH[2])
    return update_E, update_H
