from typing import Optional, Union
import numpy

from ..fdmath import dx_lists_t, fdfield_t
from ..fdmath.functional import deriv_back


# TODO documentation


def poynting(
        e: fdfield_t,
        h: fdfield_t,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Calculate the poynting vector `S` ($S$).

    This is the energy transfer rate (amount of energy `U` per `dt` transferred
    between adjacent cells) in each direction that happens during the half-step
    bounded by the two provided fields.

    The returned vector field `S` is the energy flow across +x, +y, and +z
    boundaries of the corresponding `U` cell. For example,

    ```
        mx = numpy.roll(mask, -1, axis=0)
        my = numpy.roll(mask, -1, axis=1)
        mz = numpy.roll(mask, -1, axis=2)

        u_hstep = fdtd.energy_hstep(e0=es[ii - 1], h1=hs[ii], e2=es[ii],     **args)
        u_estep = fdtd.energy_estep(h0=hs[ii],     e1=es[ii], h2=hs[ii + 1], **args)
        delta_j_B = fdtd.delta_energy_j(j0=js[ii], e1=es[ii], dxes=dxes)
        du_half_h2e = u_estep - u_hstep - delta_j_B

        s_h2e = -fdtd.poynting(e=es[ii], h=hs[ii], dxes=dxes) * dt
        planes = [s_h2e[0, mask].sum(), -s_h2e[0, mx].sum(),
                  s_h2e[1, mask].sum(), -s_h2e[1, my].sum(),
                  s_h2e[2, mask].sum(), -s_h2e[2, mz].sum()]

        assert_close(sum(planes), du_half_h2e[mask])
    ```

    (see `meanas.tests.test_fdtd.test_poynting_planes`)

    The full relationship is
    $$
      \\begin{aligned}
      (U_{l+\\frac{1}{2}} - U_l) / \\Delta_t
       &= -\\hat{\\nabla} \\cdot \\tilde{S}_{l, l + \\frac{1}{2}} \\ \\
          - \\hat{H}_{l+\\frac{1}{2}} \\cdot \\hat{M}_l \\ \\
          - \\tilde{E}_l \\cdot \\tilde{J}_{l+\\frac{1}{2}} \\\\
      (U_l - U_{l-\\frac{1}{2}}) / \\Delta_t
       &= -\\hat{\\nabla} \\cdot \\tilde{S}_{l, l - \\frac{1}{2}} \\ \\
          - \\hat{H}_{l-\\frac{1}{2}} \\cdot \\hat{M}_l \\ \\
          - \\tilde{E}_l \\cdot \\tilde{J}_{l-\\frac{1}{2}} \\\\
     \\end{aligned}
    $$

    These equalities are exact and should practically hold to within numerical precision.
    No time- or spatial-averaging is necessary. (See `meanas.fdtd` docs for derivation.)

    Args:
        e: E-field
        h: H-field (one half-timestep before or after `e`)
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        s: Vector field. Components indicate the energy transfer rate from the
            corresponding energy cell into its +x, +y, and +z neighbors during
            the half-step from the time of the earlier input field until the
            time of later input field.
    """
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


def poynting_divergence(
        s: Optional[fdfield_t] = None,
        *,
        e: Optional[fdfield_t] = None,
        h: Optional[fdfield_t] = None,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Calculate the divergence of the poynting vector.

    This is the net energy flow for each cell, i.e. the change in energy `U`
    per `dt` caused by transfer of energy to nearby cells (rather than
    absorption/emission by currents `J` or `M`).

    See `poynting` and `meanas.fdtd` for more details.
    Args:
        s: Poynting vector, as calculated with `poynting`. Optional; caller
            can provide `e` and `h` instead.
        e: E-field (optional; need either `s` or both `e` and `h`)
        h: H-field (optional; need either `s` or both `e` and `h`)
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        ds: Divergence of the poynting vector.
            Entries indicate the net energy flow out of the corresponding
            energy cell.
    """
    if s is None:
        assert(e is not None)
        assert(h is not None)
        assert(dxes is not None)
        s = poynting(e, h, dxes=dxes)

    Dx, Dy, Dz = deriv_back()
    ds = Dx(s[0]) + Dy(s[1]) + Dz(s[2])
    return ds


def energy_hstep(
        e0: fdfield_t,
        h1: fdfield_t,
        e2: fdfield_t,
        epsilon: Optional[fdfield_t] = None,
        mu: Optional[fdfield_t] = None,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Calculate energy `U` at the time of the provided H-field `h1`.

    TODO: Figure out what this means spatially.

    Args:
        e0: E-field one half-timestep before the energy.
        h1: H-field (at the same timestep as the energy).
        e2: E-field one half-timestep after the energy.
        epsilon: Dielectric constant distribution.
        mu: Magnetic permeability distribution.
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        Energy, at the time of the H-field `h1`.
    """
    u = dxmul(e0 * e2, h1 * h1, epsilon, mu, dxes)
    return u


def energy_estep(
        h0: fdfield_t,
        e1: fdfield_t,
        h2: fdfield_t,
        epsilon: Optional[fdfield_t] = None,
        mu: Optional[fdfield_t] = None,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Calculate energy `U` at the time of the provided E-field `e1`.

    TODO: Figure out what this means spatially.

    Args:
        h0: H-field one half-timestep before the energy.
        e1: E-field (at the same timestep as the energy).
        h2: H-field one half-timestep after the energy.
        epsilon: Dielectric constant distribution.
        mu: Magnetic permeability distribution.
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        Energy, at the time of the E-field `e1`.
    """
    u = dxmul(e1 * e1, h0 * h2, epsilon, mu, dxes)
    return u


def delta_energy_h2e(
        dt: float,
        e0: fdfield_t,
        h1: fdfield_t,
        e2: fdfield_t,
        h3: fdfield_t,
        epsilon: Optional[fdfield_t] = None,
        mu: Optional[fdfield_t] = None,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Change in energy during the half-step from `h1` to `e2`.

    This is just from (e2 * e2 + h3 * h1) - (h1 * h1 + e0 * e2)

    Args:
        e0: E-field one half-timestep before the start of the energy delta.
        h1: H-field at the start of the energy delta.
        e2: E-field at the end of the energy delta (one half-timestep after `h1`).
        h3: H-field one half-timestep after the end of the energy delta.
        epsilon: Dielectric constant distribution.
        mu: Magnetic permeability distribution.
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        Change in energy from the time of `h1` to the time of `e2`.
    """
    de = e2 * (e2 - e0) / dt
    dh = h1 * (h3 - h1) / dt
    du = dxmul(de, dh, epsilon, mu, dxes)
    return du


def delta_energy_e2h(
        dt: float,
        h0: fdfield_t,
        e1: fdfield_t,
        h2: fdfield_t,
        e3: fdfield_t,
        epsilon: Optional[fdfield_t] = None,
        mu: Optional[fdfield_t] = None,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Change in energy during the half-step from `e1` to `h2`.

    This is just from (h2 * h2 + e3 * e1) - (e1 * e1 + h0 * h2)

    Args:
        h0: E-field one half-timestep before the start of the energy delta.
        e1: H-field at the start of the energy delta.
        h2: E-field at the end of the energy delta (one half-timestep after `e1`).
        e3: H-field one half-timestep after the end of the energy delta.
        epsilon: Dielectric constant distribution.
        mu: Magnetic permeability distribution.
        dxes: Grid description; see `meanas.fdmath`.

    Returns:
        Change in energy from the time of `e1` to the time of `h2`.
    """
    de = e1 * (e3 - e1) / dt
    dh = h2 * (h2 - h0) / dt
    du = dxmul(de, dh, epsilon, mu, dxes)
    return du


def delta_energy_j(
        j0: fdfield_t,
        e1: fdfield_t,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    """
    Calculate

    Note that each value of $J$ contributes to the energy twice (i.e. once per field update)
    despite only causing the value of $E$ to change once (same for $M$ and $H$).


    """
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    du = ((j0 * e1).sum(axis=0)
          * dxes[0][0][:, None, None]
          * dxes[0][1][None, :, None]
          * dxes[0][2][None, None, :])
    return du


def dxmul(
        ee: fdfield_t,
        hh: fdfield_t,
        epsilon: Optional[Union[fdfield_t, float]] = None,
        mu: Optional[Union[fdfield_t, float]] = None,
        dxes: Optional[dx_lists_t] = None,
        ) -> fdfield_t:
    if epsilon is None:
        epsilon = 1
    if mu is None:
        mu = 1
    if dxes is None:
        dxes = tuple(tuple(numpy.ones(1) for _ in range(3)) for _ in range(2))

    result = ((ee * epsilon).sum(axis=0)
              * dxes[0][0][:, None, None]
              * dxes[0][1][None, :, None]
              * dxes[0][2][None, None, :]
              + (hh * mu).sum(axis=0)
              * dxes[1][0][:, None, None]
              * dxes[1][1][None, :, None]
              * dxes[1][2][None, None, :])
    return result
