"""
Tools for working with waveguide modes in 3D domains.

This module relies heavily on `waveguide_2d` and mostly just transforms
its parameters into 2D equivalents and expands the results back into 3D.
"""
from typing import Dict, Optional, Sequence, Union, Any
import numpy
from numpy.typing import NDArray

from ..fdmath import vec, unvec, dx_lists_t, fdfield_t, cfdfield_t
from . import operators, waveguide_2d


def solve_mode(
        mode_number: int,
        omega: complex,
        dxes: dx_lists_t,
        axis: int,
        polarity: int,
        slices: Sequence[slice],
        epsilon: fdfield_t,
        mu: Optional[fdfield_t] = None,
        ) -> Dict[str, Union[complex, NDArray[numpy.float_]]]:
    """
    Given a 3D grid, selects a slice from the grid and attempts to
     solve for an eigenmode propagating through that slice.

    Args:
        mode_number: Number of the mode, 0-indexed
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        axis: Propagation axis (0=x, 1=y, 2=z)
        polarity: Propagation direction (+1 for +ve, -1 for -ve)
        slices: `epsilon[tuple(slices)]` is used to select the portion of the grid to use
                as the waveguide cross-section. `slices[axis]` should select only one item.
        epsilon: Dielectric constant
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        ```
        {
            'E': List[NDArray[numpy.float_]],
            'H': List[NDArray[numpy.float_]],
            'wavenumber': complex,
        }
        ```
    """
    if mu is None:
        mu = numpy.ones_like(epsilon)

    slices = tuple(slices)

    '''
    Solve the 2D problem in the specified plane
    '''
    # Define rotation to set z as propagation direction
    order = numpy.roll(range(3), 2 - axis)
    reverse_order = numpy.roll(range(3), axis - 2)

    # Find dx in propagation direction
    dxab_forward = numpy.array([dx[order[2]][slices[order[2]]] for dx in dxes])
    dx_prop = 0.5 * dxab_forward.sum()

    # Reduce to 2D and solve the 2D problem
    args_2d: Dict[str, Any] = {
        'omega': omega,
        'dxes': [[dx[i][slices[i]] for i in order[:2]] for dx in dxes],
        'epsilon': vec([epsilon[i][slices].transpose(order) for i in order]),
        'mu': vec([mu[i][slices].transpose(order) for i in order]),
    }
    e_xy, wavenumber_2d = waveguide_2d.solve_mode(mode_number, **args_2d)

    '''
    Apply corrections and expand to 3D
    '''
    # Correct wavenumber to account for numerical dispersion.
    wavenumber = 2 / dx_prop * numpy.arcsin(wavenumber_2d * dx_prop / 2)

    shape = [d.size for d in args_2d['dxes'][0]]
    ve, vh = waveguide_2d.normalized_fields_e(e_xy, wavenumber=wavenumber_2d, prop_phase=dx_prop * wavenumber, **args_2d)
    e = unvec(ve, shape)
    h = unvec(vh, shape)

    # Adjust for propagation direction
    h *= polarity           # type: ignore          # mypy issue with numpy

    # Apply phase shift to H-field
    h[:2] *= numpy.exp(-1j * polarity * 0.5 * wavenumber * dx_prop)
    e[2]  *= numpy.exp(-1j * polarity * 0.5 * wavenumber * dx_prop)

    # Expand E, H to full epsilon space we were given
    E = numpy.zeros_like(epsilon, dtype=complex)
    H = numpy.zeros_like(epsilon, dtype=complex)
    for a, o in enumerate(reverse_order):
        E[(a, *slices)] = e[o][:, :, None].transpose(reverse_order)
        H[(a, *slices)] = h[o][:, :, None].transpose(reverse_order)

    results = {
        'wavenumber': wavenumber,
        'wavenumber_2d': wavenumber_2d,
        'H': H,
        'E': E,
    }
    return results


def compute_source(
        E: cfdfield_t,
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        axis: int,
        polarity: int,
        slices: Sequence[slice],
        epsilon: fdfield_t,
        mu: Optional[fdfield_t] = None,
        ) -> cfdfield_t:
    """
    Given an eigenmode obtained by `solve_mode`, returns the current source distribution
    necessary to position a unidirectional source at the slice location.

    Args:
        E: E-field of the mode
        wavenumber: Wavenumber of the mode
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        axis: Propagation axis (0=x, 1=y, 2=z)
        polarity: Propagation direction (+1 for +ve, -1 for -ve)
        slices: `epsilon[tuple(slices)]` is used to select the portion of the grid to use
                as the waveguide cross-section. `slices[axis]` should select only one item.
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        J distribution for the unidirectional source
    """
    E_expanded = expand_e(E=E, dxes=dxes, wavenumber=wavenumber, axis=axis,
                          polarity=polarity, slices=slices)

    smask = [slice(None)] * 4
    if polarity > 0:
        smask[axis + 1] = slice(slices[axis].start, None)
    else:
        smask[axis + 1] = slice(None, slices[axis].stop)

    mask = numpy.zeros_like(E_expanded, dtype=int)
    mask[tuple(smask)] = 1

    masked_e2j = operators.e_boundary_source(mask=vec(mask), omega=omega, dxes=dxes, epsilon=vec(epsilon), mu=vec(mu))
    J = unvec(masked_e2j @ vec(E_expanded), E.shape[1:])
    return J


def compute_overlap_e(
        E: cfdfield_t,
        wavenumber: complex,
        dxes: dx_lists_t,
        axis: int,
        polarity: int,
        slices: Sequence[slice],
        ) -> cfdfield_t:                 # TODO DOCS
    """
    Given an eigenmode obtained by `solve_mode`, calculates an overlap_e for the
    mode orthogonality relation Integrate(((E x H_mode) + (E_mode x H)) dot dn)
    [assumes reflection symmetry].

    TODO: add reference

    Args:
        E: E-field of the mode
        H: H-field of the mode (advanced by half of a Yee cell from E)
        wavenumber: Wavenumber of the mode
        omega: Angular frequency of the simulation
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        axis: Propagation axis (0=x, 1=y, 2=z)
        polarity: Propagation direction (+1 for +ve, -1 for -ve)
        slices: `epsilon[tuple(slices)]` is used to select the portion of the grid to use
                as the waveguide cross-section. slices[axis] should select only one item.
        mu: Magnetic permeability (default 1 everywhere)

    Returns:
        overlap_e such that `numpy.sum(overlap_e * other_e.conj())` computes the overlap integral
    """
    slices = tuple(slices)

    Ee = expand_e(E=E, wavenumber=wavenumber, dxes=dxes,
                  axis=axis, polarity=polarity, slices=slices)

    start, stop = sorted((slices[axis].start, slices[axis].start - 2 * polarity))

    slices2_l = list(slices)
    slices2_l[axis] = slice(start, stop)
    slices2 = (slice(None), *slices2_l)

    Etgt = numpy.zeros_like(Ee)
    Etgt[slices2] = Ee[slices2]

    Etgt /= (Etgt.conj() * Etgt).sum()
    return Etgt


def expand_e(
        E: cfdfield_t,
        wavenumber: complex,
        dxes: dx_lists_t,
        axis: int,
        polarity: int,
        slices: Sequence[slice],
        ) -> cfdfield_t:
    """
    Given an eigenmode obtained by `solve_mode`, expands the E-field from the 2D
    slice where the mode was calculated to the entire domain (along the propagation
    axis). This assumes the epsilon cross-section remains constant throughout the
    entire domain; it is up to the caller to truncate the expansion to any regions
    where it is valid.

    Args:
        E: E-field of the mode
        wavenumber: Wavenumber of the mode
        dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
        axis: Propagation axis (0=x, 1=y, 2=z)
        polarity: Propagation direction (+1 for +ve, -1 for -ve)
        slices: `epsilon[tuple(slices)]` is used to select the portion of the grid to use
                as the waveguide cross-section. slices[axis] should select only one item.

    Returns:
        `E`, with the original field expanded along the specified `axis`.
    """
    slices = tuple(slices)

    # Determine phase factors for parallel slices
    a_shape = numpy.roll([1, -1, 1, 1], axis)
    a_E = numpy.real(dxes[0][axis]).cumsum()
    r_E = a_E - a_E[slices[axis]]
    iphi = polarity * -1j * wavenumber
    phase_E = numpy.exp(iphi * r_E).reshape(a_shape)

    # Expand our slice to the entire grid using the phase factors
    E_expanded = numpy.zeros_like(E)

    slices_exp_l = list(slices)
    slices_exp_l[axis] = slice(E.shape[axis + 1])
    slices_exp = (slice(None), *slices_exp_l)

    slices_in = (slice(None), *slices)

    E_expanded[slices_exp] = phase_E * numpy.array(E)[slices_in]
    return E_expanded
