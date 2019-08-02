from typing import Dict, List
import numpy
import scipy.sparse as sparse

from . import vec, unvec, dx_lists_t, vfield_t, field_t
from . import operators, waveguide, functional
from .eigensolvers import signed_eigensolve, rayleigh_quotient_iteration


def solve_waveguide_mode_2d(mode_number: int,
                            omega: complex,
                            dxes: dx_lists_t,
                            epsilon: vfield_t,
                            mu: vfield_t = None,
                            wavenumber_correction: bool = True,
                            ) -> Dict[str, complex or field_t]:
    """
    Given a 2d region, attempts to solve for the eigenmode with the specified mode number.

    :param mode_number: Number of the mode, 0-indexed.
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :param wavenumber_correction: Whether to correct the wavenumber to
        account for numerical dispersion (default True)
    :return: {'E': List[numpy.ndarray], 'H': List[numpy.ndarray], 'wavenumber': complex}
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
    '''
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]
    A_r = waveguide.operator(numpy.real(omega), dxes_real, numpy.real(epsilon), numpy.real(mu))

    eigvals, eigvecs = signed_eigensolve(A_r, mode_number+3)
    v = eigvecs[:, -(mode_number + 1)]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = waveguide.operator(omega, dxes, epsilon, mu)
    eigval, v = rayleigh_quotient_iteration(A, v)

    # Calculate the wave-vector (force the real part to be positive)
    wavenumber = numpy.sqrt(eigval)
    wavenumber *= numpy.sign(numpy.real(wavenumber))

    e, h = waveguide.normalized_fields(v, wavenumber, omega, dxes, epsilon, mu)

    '''
    Perform correction on wavenumber to account for numerical dispersion.

     See Numerical Dispersion in Taflove's FDTD book.
     This correction term reduces the error in emitted power, but additional
      error is introduced into the E_err and H_err terms. This effect becomes
      more pronounced as the wavenumber increases.
    '''
    if wavenumber_correction:
        dx_mean = (numpy.hstack(dxes[0]) + numpy.hstack(dxes[1])).mean() / 2        #TODO figure out what dx to use here
        wavenumber -= 2 * numpy.sin(numpy.real(wavenumber * dx_mean / 2)) / dx_mean - numpy.real(wavenumber)

    shape = [d.size for d in dxes[0]]
    fields = {
        'wavenumber': wavenumber,
        'E': unvec(e, shape),
        'H': unvec(h, shape),
    }

    return fields


def solve_waveguide_mode(mode_number: int,
                         omega: complex,
                         dxes: dx_lists_t,
                         axis: int,
                         polarity: int,
                         slices: List[slice],
                         epsilon: field_t,
                         mu: field_t = None,
                         wavenumber_correction: bool = True
                         ) -> Dict[str, complex or numpy.ndarray]:
    """
    Given a 3D grid, selects a slice from the grid and attempts to
     solve for an eigenmode propagating through that slice.

    :param mode_number: Number of the mode, 0-indexed
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :param wavenumber_correction: Whether to correct the wavenumber to
        account for numerical dispersion (default True)
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

    # Reduce to 2D and solve the 2D problem
    args_2d = {
        'dxes': [[dx[i][slices[i]] for i in order[:2]] for dx in dxes],
        'epsilon': vec([epsilon[i][slices].transpose(order) for i in order]),
        'mu': vec([mu[i][slices].transpose(order) for i in order]),
        'wavenumber_correction': wavenumber_correction,
    }
    fields_2d = solve_waveguide_mode_2d(mode_number, omega=omega, **args_2d)

    '''
    Apply corrections and expand to 3D
    '''
    # Scale based on dx in propagation direction
    dxab_forward = numpy.array([dx[order[2]][slices[order[2]]] for dx in dxes])

    # Adjust for propagation direction
    fields_2d['E'][2] *= polarity
    fields_2d['H'][2] *= polarity

    # Apply phase shift to H-field
    d_prop = 0.5 * sum(dxab_forward)
    fields_2d['H'] *= numpy.exp(-polarity * 1j * 0.5 * fields_2d['wavenumber'] * d_prop)

    # Expand E, H to full epsilon space we were given
    E = numpy.zeros_like(epsilon, dtype=complex)
    H = numpy.zeros_like(epsilon, dtype=complex)
    for a, o in enumerate(reverse_order):
        E[(a, *slices)] = fields_2d['E'][o][:, :, None].transpose(reverse_order)
        H[(a, *slices)] = fields_2d['H'][o][:, :, None].transpose(reverse_order)

    results = {
        'wavenumber': fields_2d['wavenumber'],
        'H': H,
        'E': E,
    }

    return results


def compute_source(E: field_t,
                   H: field_t,
                   wavenumber: complex,
                   omega: complex,
                   dxes: dx_lists_t,
                   axis: int,
                   polarity: int,
                   slices: List[slice],
                   mu: field_t = None,
                   ) -> field_t:
    """
    Given an eigenmode obtained by solve_waveguide_mode, returns the current source distribution
    necessary to position a unidirectional source at the slice location.

    :param E: E-field of the mode
    :param H: H-field of the mode (advanced by half of a Yee cell from E)
    :param wavenumber: Wavenumber of the mode
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: J distribution for the unidirectional source
    """
    if mu is None:
        mu = numpy.ones(3)

    J = numpy.zeros_like(E, dtype=complex)
    M = numpy.zeros_like(E, dtype=complex)

    src_order = numpy.roll(range(3), -axis)
    exp_iphi = numpy.exp(1j * polarity * wavenumber * dxes[1][axis][slices[axis]])
    J[src_order[1]] = +exp_iphi * H[src_order[2]] * polarity
    J[src_order[2]] = -exp_iphi * H[src_order[1]] * polarity

    rollby = -1 if polarity > 0 else 0
    M[src_order[1]] = +numpy.roll(E[src_order[2]], rollby, axis=axis)
    M[src_order[2]] = -numpy.roll(E[src_order[1]], rollby, axis=axis)

    m2j = functional.m2j(omega, dxes, mu)
    Jm = m2j(M)

    Jtot = J + Jm
    return Jtot


def compute_overlap_e(E: field_t,
                      H: field_t,
                      wavenumber: complex,
                      omega: complex,
                      dxes: dx_lists_t,
                      axis: int,
                      polarity: int,
                      slices: List[slice],
                      mu: field_t = None,
                      ) -> field_t:
    """
    Given an eigenmode obtained by solve_waveguide_mode, calculates overlap_e for the
    mode orthogonality relation Integrate(((E x H_mode) + (E_mode x H)) dot dn)
    [assumes reflection symmetry].

    overlap_e makes use of the e2h operator to collapse the above expression into
     (vec(E) @ vec(overlap_e)), allowing for simple calculation of the mode overlap.

    :param E: E-field of the mode
    :param H: H-field of the mode (advanced by half of a Yee cell from E)
    :param wavenumber: Wavenumber of the mode
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: overlap_e for calculating the mode overlap
    """
    slices = tuple(slices)

    cross_plane = [slice(None)] * 4
    cross_plane[axis + 1] = slices[axis]
    cross_plane = tuple(cross_plane)

    # Determine phase factors for parallel slices
    a_shape = numpy.roll([-1, 1, 1], axis)
    a_E = numpy.real(dxes[0][axis]).cumsum()
    a_H = numpy.real(dxes[1][axis]).cumsum()
    iphi = -polarity * 1j * wavenumber
    phase_E = numpy.exp(iphi * (a_E - a_E[slices[axis]])).reshape(a_shape)
    phase_H = numpy.exp(iphi * (a_H - a_H[slices[axis]])).reshape(a_shape)

    # Expand our slice to the entire grid using the calculated phase factors
    Ee = phase_E * E[cross_plane]
    He = phase_H * H[cross_plane]


    # Write out the operator product for the mode orthogonality integral
    domain = numpy.zeros_like(E[0], dtype=int)
    domain[slices] = 1

    npts = E[0].size
    dn = numpy.zeros(npts * 3, dtype=int)
    dn[0:npts] = 1
    dn = numpy.roll(dn, npts * axis)

    e2h = operators.e2h(omega, dxes, mu)
    ds = sparse.diags(vec([domain]*3))
    h_cross_ = operators.poynting_h_cross(vec(He), dxes)
    e_cross_ = operators.poynting_e_cross(vec(Ee), dxes)

    overlap_e = dn @ ds @ (-h_cross_ + e_cross_ @ e2h)

    # Normalize
    dx_forward = dxes[0][axis][slices[axis]]
    norm_factor = numpy.abs(overlap_e @ vec(Ee))
    overlap_e /= norm_factor * dx_forward

    return unvec(overlap_e, E[0].shape)


def solve_waveguide_mode_cylindrical(mode_number: int,
                                     omega: complex,
                                     dxes: dx_lists_t,
                                     epsilon: vfield_t,
                                     r0: float,
                                     wavenumber_correction: bool = True,
                                     ) -> Dict[str, complex or field_t]:
    """
    Given a 2d (r, y) slice of epsilon, attempts to solve for the eigenmode
     of the bent waveguide with the specified mode number.

    :param mode_number: Number of the mode, 0-indexed
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header.
        The first coordinate is assumed to be r, the second is y.
    :param epsilon: Dielectric constant
    :param r0: Radius of curvature for the simulation. This should be the minimum value of
        r within the simulation domain.
    :param wavenumber_correction: Whether to correct the wavenumber to
        account for numerical dispersion (default True)
    :return: {'E': List[numpy.ndarray], 'H': List[numpy.ndarray], 'wavenumber': complex}
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
    '''
    dxes_real = [[numpy.real(dx) for dx in dxi] for dxi in dxes]

    A_r = waveguide.cylindrical_operator(numpy.real(omega), dxes_real, numpy.real(epsilon), r0)
    eigvals, eigvecs = signed_eigensolve(A_r, mode_number + 3)
    v = eigvecs[:, -(mode_number+1)]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = waveguide.cylindrical_operator(omega, dxes, epsilon, r0)
    eigval, v = rayleigh_quotient_iteration(A, v)

    # Calculate the wave-vector (force the real part to be positive)
    wavenumber = numpy.sqrt(eigval)
    wavenumber *= numpy.sign(numpy.real(wavenumber))

    '''
    Perform correction on wavenumber to account for numerical dispersion.

     See Numerical Dispersion in Taflove's FDTD book.
     This correction term reduces the error in emitted power, but additional
      error is introduced into the E_err and H_err terms. This effect becomes
      more pronounced as the wavenumber increases.
    '''
    if wavenumber_correction:
        wavenumber -= 2 * numpy.sin(numpy.real(wavenumber / 2)) - numpy.real(wavenumber)

    shape = [d.size for d in dxes[0]]
    v = numpy.hstack((v, numpy.zeros(shape[0] * shape[1])))
    fields = {
        'wavenumber': wavenumber,
        'E': unvec(v, shape),
#        'E': unvec(e, shape),
#        'H': unvec(h, shape),
    }

    return fields


def compute_source_q(E: field_t,
                     H: field_t,
                     wavenumber: complex,
                     omega: complex,
                     dxes: dx_lists_t,
                     axis: int,
                     polarity: int,
                     slices: List[slice],
                     mu: field_t = None,
                     ) -> field_t:
    A1f = functional.curl_h(dxes)
    A2f = functional.curl_e(dxes)

    J = A1f(H)
    M = A2f(-E)

    m2j = functional.m2j(omega, dxes, mu)
    Jm = m2j(M)

    Jtot = J + Jm

    return Jtot, J, M


def compute_source_e(QE: field_t,
                     omega: complex,
                     dxes: dx_lists_t,
                     axis: int,
                     polarity: int,
                     slices: List[slice],
                     epsilon: field_t,
                     mu: field_t = None,
                     ) -> field_t:
    """
    Want (AQ-QA) E = -iwj, where Q is a mask
    If E is an eigenmode, AE = 0 so just AQE = -iwJ
    Really only need E in 4 cells along axis (0, 0, Emode1, Emode2), find AE (1 fdtd step), then use center 2 cells as src
    """
    slices = tuple(slices)

    # Trim a cell from each end of the propagation axis
    slices_reduced = list(slices)
    slices_reduced[axis] = slice(slices[axis].start + 1, slices[axis].stop - 1)
    slices_reduced = tuple(slice(None), *slices)

    # Don't actually need to mask out E here since it needs to be pre-masked (QE)

    A = functional.e_full(omega, dxes, epsilon, mu)
    J4 = A(QE) / (-1j * omega)             #J4 is 4-cell result of -iwJ = A QE

    J = numpy.zeros_like(J4)
    J[slices_reduced] = J4[slices_reduced]
    return J


def compute_source_wg(E: field_t,
                      wavenumber: complex,
                      omega: complex,
                      dxes: dx_lists_t,
                      axis: int,
                      polarity: int,
                      slices: List[slice],
                      epsilon: field_t,
                      mu: field_t = None,
                      ) -> field_t:
    slices = tuple(slices)
    Etgt, _slices2 = compute_overlap_ce(E=E, wavenumber=wavenumber,
                                        dxes=dxes, axis=axis, polarity=polarity,
                                        slices=slices)

    slices4 = list(slices)
    slices4[axis] = slice(slices[axis].start - 4 * polarity, slices[axis].start)
    slices4 = tuple(slices4)

    J = compute_source_e(QE=Etgt,
                         omega=omega, dxes=dxes, axis=axis,
                         polarity=polarity, slices=slices4,
                         epsilon=epsilon, mu=mu)
    return J


def compute_overlap_ce(E: field_t,
                       wavenumber: complex,
                       dxes: dx_lists_t,
                       axis: int,
                       polarity: int,
                       slices: List[slice],
                       ) -> field_t:
    slices = tuple(slices)

    Ee = expand_wgmode_e(E=E, wavenumber=wavenumber,
                         dxes=dxes, axis=axis, polarity=polarity,
                         slices=slices)

    start, stop = sorted((slices[axis].start, slices[axis].start - 2 * polarity))

    slices2 = list(slices)
    slices2[axis] = slice(start, stop)
    slices2 = tuple(slice(None), slices2)

    Etgt = numpy.zeros_like(Ee)
    Etgt[slices2] = Ee[slices2]

    Etgt /= (Etgt.conj() * Etgt).sum()
    return Etgt, slices2


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
    iphi = polarity * 1j * wavenumber
    phase_E = numpy.exp(iphi * r_E).reshape(a_shape)

    # Expand our slice to the entire grid using the phase factors
    Ee = numpy.zeros_like(E)

    slices_exp = list(slices)
    slices_exp[axis] = slice(E.shape[axis + 1])
    slices_exp = (slice(None), *slices_exp)

    slices_in = tuple(slice(None), *slices)

    Ee[slices_exp] = phase_E * numpy.array(E)[slices_in]

    return Ee


