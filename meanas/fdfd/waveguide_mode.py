from typing import Dict, List, Tuple
import numpy
import scipy.sparse as sparse

from .. import vec, unvec, dx_lists_t, vfield_t, field_t
from . import operators, waveguide, functional
from ..eigensolvers import signed_eigensolve, rayleigh_quotient_iteration


def vsolve_waveguide_mode_2d(mode_number: int,
                             omega: complex,
                             dxes: dx_lists_t,
                             epsilon: vfield_t,
                             mu: vfield_t = None,
                             mode_margin: int = 2,
                             ) -> Tuple[vfield_t, complex]:
    """
    Given a 2d region, attempts to solve for the eigenmode with the specified mode number.

    :param mode_number: Number of the mode, 0-indexed.
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :param mode_margin: The eigensolver will actually solve for (mode_number + mode_margin)
        modes, but only return the target mode. Increasing this value can improve the solver's
        ability to find the correct mode. Default 2.
    :return: (e_xy, wavenumber)
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
    '''
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]
    A_r = waveguide.operator_e(numpy.real(omega), dxes_real, numpy.real(epsilon), numpy.real(mu))

    eigvals, eigvecs = signed_eigensolve(A_r, mode_number + mode_margin)
    e_xy = eigvecs[:, -(mode_number + 1)]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = waveguide.operator_e(omega, dxes, epsilon, mu)
    eigval, e_xy = rayleigh_quotient_iteration(A, e_xy)

    # Calculate the wave-vector (force the real part to be positive)
    wavenumber = numpy.sqrt(eigval)
    wavenumber *= numpy.sign(numpy.real(wavenumber))

    return e_xy, wavenumber



def solve_waveguide_mode(mode_number: int,
                         omega: complex,
                         dxes: dx_lists_t,
                         axis: int,
                         polarity: int,
                         slices: List[slice],
                         epsilon: field_t,
                         mu: field_t = None,
                         ) -> Dict[str, complex or numpy.ndarray]:
    """
    Given a 3D grid, selects a slice from the grid and attempts to
     solve for an eigenmode propagating through that slice.

    :param mode_number: Number of the mode, 0-indexed
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: {'E': List[numpy.ndarray], 'H': List[numpy.ndarray], 'wavenumber': complex}
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
    dx_prop = 0.5 * sum(dxab_forward)[0]

    # Reduce to 2D and solve the 2D problem
    args_2d = {
        'omega': omega,
        'dxes': [[dx[i][slices[i]] for i in order[:2]] for dx in dxes],
        'epsilon': vec([epsilon[i][slices].transpose(order) for i in order]),
        'mu': vec([mu[i][slices].transpose(order) for i in order]),
    }
    e_xy, wavenumber_2d = vsolve_waveguide_mode_2d(mode_number, **args_2d)

    '''
    Apply corrections and expand to 3D
    '''
    # Correct wavenumber to account for numerical dispersion.
    wavenumber = 2/dx_prop * numpy.arcsin(wavenumber_2d * dx_prop/2)
    print(wavenumber_2d / wavenumber)

    shape = [d.size for d in args_2d['dxes'][0]]
    ve, vh = waveguide.normalized_fields_e(e_xy, wavenumber=wavenumber_2d, **args_2d, prop_phase=dx_prop * wavenumber)
    e = unvec(ve, shape)
    h = unvec(vh, shape)

    # Adjust for propagation direction
    h *= polarity

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


def compute_source(E: field_t,
                   wavenumber: complex,
                   omega: complex,
                   dxes: dx_lists_t,
                   axis: int,
                   polarity: int,
                   slices: List[slice],
                   epsilon: field_t,
                   mu: field_t = None,
                   ) -> field_t:
    """
    Given an eigenmode obtained by solve_waveguide_mode, returns the current source distribution
    necessary to position a unidirectional source at the slice location.

    :param E: E-field of the mode
    :param wavenumber: Wavenumber of the mode
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: J distribution for the unidirectional source
    """
    E_expanded = expand_wgmode_e(E=E, dxes=dxes, wavenumber=wavenumber, axis=axis,
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


def compute_overlap_e(E: field_t,
                      wavenumber: complex,
                      dxes: dx_lists_t,
                      axis: int,
                      polarity: int,
                      slices: List[slice],
                      ) -> field_t:                 # TODO DOCS
    """
    Given an eigenmode obtained by solve_waveguide_mode, calculates overlap_e for the
    mode orthogonality relation Integrate(((E x H_mode) + (E_mode x H)) dot dn)
    [assumes reflection symmetry].i

    overlap_e makes use of the e2h operator to collapse the above expression into
     (vec(E) @ vec(overlap_e)), allowing for simple calculation of the mode overlap.

    :param E: E-field of the mode
    :param H: H-field of the mode (advanced by half of a Yee cell from E)
    :param wavenumber: Wavenumber of the mode
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: overlap_e for calculating the mode overlap
    """
    slices = tuple(slices)

    Ee = expand_wgmode_e(E=E, wavenumber=wavenumber, dxes=dxes,
                         axis=axis, polarity=polarity, slices=slices)

    start, stop = sorted((slices[axis].start, slices[axis].start - 2 * polarity))

    slices2 = list(slices)
    slices2[axis] = slice(start, stop)
    slices2 = (slice(None), *slices2)

    Etgt = numpy.zeros_like(Ee)
    Etgt[slices2] = Ee[slices2]

    Etgt /= (Etgt.conj() * Etgt).sum()
    return Etgt


def solve_waveguide_mode_cylindrical(mode_number: int,
                                     omega: complex,
                                     dxes: dx_lists_t,
                                     epsilon: vfield_t,
                                     r0: float,
                                     ) -> Dict[str, complex or field_t]:
    """
    TODO: fixup
    Given a 2d (r, y) slice of epsilon, attempts to solve for the eigenmode
     of the bent waveguide with the specified mode number.

    :param mode_number: Number of the mode, 0-indexed
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in meanas.types.
        The first coordinate is assumed to be r, the second is y.
    :param epsilon: Dielectric constant
    :param r0: Radius of curvature for the simulation. This should be the minimum value of
        r within the simulation domain.
    :return: {'E': List[numpy.ndarray], 'H': List[numpy.ndarray], 'wavenumber': complex}
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
    '''
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]

    A_r = waveguide.cylindrical_operator(numpy.real(omega), dxes_real, numpy.real(epsilon), r0)
    eigvals, eigvecs = signed_eigensolve(A_r, mode_number + 3)
    e_xy = eigvecs[:, -(mode_number+1)]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = waveguide.cylindrical_operator(omega, dxes, epsilon, r0)
    eigval, e_xy = rayleigh_quotient_iteration(A, e_xy)

    # Calculate the wave-vector (force the real part to be positive)
    wavenumber = numpy.sqrt(eigval)
    wavenumber *= numpy.sign(numpy.real(wavenumber))

    # TODO: Perform correction on wavenumber to account for numerical dispersion.

    shape = [d.size for d in dxes[0]]
    e_xy = numpy.hstack((e_xy, numpy.zeros(shape[0] * shape[1])))
    fields = {
        'wavenumber': wavenumber,
        'E': unvec(e_xy, shape),
#        'E': unvec(e, shape),
#        'H': unvec(h, shape),
    }

    return fields


def expand_wgmode_e(E: field_t,
                    wavenumber: complex,
                    dxes: dx_lists_t,
                    axis: int,
                    polarity: int,
                    slices: List[slice],
                    ) -> field_t:
    slices = tuple(slices)

    # Determine phase factors for parallel slices
    a_shape = numpy.roll([1, -1, 1, 1], axis)
    a_E = numpy.real(dxes[0][axis]).cumsum()
    r_E = a_E - a_E[slices[axis]]
    iphi = polarity * -1j * wavenumber
    phase_E = numpy.exp(iphi * r_E).reshape(a_shape)

    # Expand our slice to the entire grid using the phase factors
    E_expanded = numpy.zeros_like(E)

    slices_exp = list(slices)
    slices_exp[axis] = slice(E.shape[axis + 1])
    slices_exp = (slice(None), *slices_exp)

    slices_in = (slice(None), *slices)

    E_expanded[slices_exp] = phase_E * numpy.array(E)[slices_in]
    return E_expanded


