"""
Basic FDTD field updates


"""
from typing import Union

from ..fdmath import dx_lists_t, fdfield_t, fdfield_updater_t
from ..fdmath.functional import curl_forward, curl_back


__author__ = 'Jan Petykiewicz'


def maxwell_e(dt: float, dxes: dx_lists_t = None) -> fdfield_updater_t:
    """
    Build a function which performs a portion the time-domain E-field update,

        E += curl_back(H[t]) / epsilon

    The full update should be

        E += (curl_back(H[t]) + J) / epsilon

    which requires an additional step of `E += J / epsilon` which is not performed
    by the generated function.

    See `meanas.fdmath` for descriptions of

    - This update step: "Maxwell's equations" section
    - `dxes`: "Datastructure: dx_lists_t" section
    - `epsilon`: "Permittivity and Permeability" section

    Also see the "Timestep" section of `meanas.fdtd` for a discussion of
    the `dt` parameter.

    Args:
        dt: Timestep. See `meanas.fdtd` for details.
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        Function `f(E_old, H_old, epsilon) -> E_new`.
    """
    if dxes is not None:
        curl_h_fun = curl_back(dxes[1])
    else:
        curl_h_fun = curl_back()

    def me_fun(e: fdfield_t, h: fdfield_t, epsilon: Union[fdfield_t, float]) -> fdfield_t:
        """
        Update the E-field.

        Args:
            e: E-field at time t=0
            h: H-field at time t=0.5
            epsilon: Dielectric constant distribution.

        Returns:
            E-field at time t=1
        """
        e += dt * curl_h_fun(h) / epsilon           # type: ignore          # mypy gets confused around ndarray ops
        return e

    return me_fun


def maxwell_h(dt: float, dxes: dx_lists_t = None) -> fdfield_updater_t:
    """
    Build a function which performs part of the time-domain H-field update,

        H -= curl_forward(E[t]) / mu

    The full update should be

        H -= (curl_forward(E[t]) + M) / mu

    which requires an additional step of `H -= M / mu` which is not performed
    by the generated function; this step can be omitted if there is no magnetic
    current `M`.

    See `meanas.fdmath` for descriptions of

    - This update step: "Maxwell's equations" section
    - `dxes`: "Datastructure: dx_lists_t" section
    - `mu`: "Permittivity and Permeability" section

    Also see the "Timestep" section of `meanas.fdtd` for a discussion of
    the `dt` parameter.

    Args:
        dt: Timestep. See `meanas.fdtd` for details.
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        Function `f(E_old, H_old, epsilon) -> E_new`.
    """
    if dxes is not None:
        curl_e_fun = curl_forward(dxes[0])
    else:
        curl_e_fun = curl_forward()

    def mh_fun(e: fdfield_t, h: fdfield_t, mu: Union[fdfield_t, float, None] = None) -> fdfield_t:
        """
        Update the H-field.

        Args:
            e: E-field at time t=1
            h: H-field at time t=0.5
            mu: Magnetic permeability. Default is 1 everywhere.

        Returns:
            H-field at time t=1.5
        """
        if mu is not None:
            h -= dt * curl_e_fun(e) / mu           # type: ignore          # mypy gets confused around ndarray ops
        else:
            h -= dt * curl_e_fun(e)                # type: ignore          # mypy gets confused around ndarray ops

        return h

    return mh_fun
