"""
Functions for performing near-to-farfield transformation (and the reverse).
"""
from typing import Dict, List, Any
import numpy
from numpy.fft import fft2, fftshift, fftfreq, ifft2, ifftshift
from numpy import pi

from ..fdmath import cfdfield_t


def near_to_farfield(
        E_near: cfdfield_t,
        H_near: cfdfield_t,
        dx: float,
        dy: float,
        padded_size: List[int] = None
        ) -> Dict[str, Any]:
    """
    Compute the farfield, i.e. the distribution of the fields after propagation
      through several wavelengths of uniform medium.

    The input fields should be complex phasors.

    Args:
        E_near: List of 2 ndarrays containing the 2D phasor field slices for the transverse
                E fields (e.g. [Ex, Ey] for calculating the farfield toward the z-direction).
        H_near: List of 2 ndarrays containing the 2D phasor field slices for the transverse
                H fields (e.g. [Hx, hy] for calculating the farfield towrad the z-direction).
        dx: Cell size along x-dimension, in units of wavelength.
        dy: Cell size along y-dimension, in units of wavelength.
        padded_size: Shape of the output. A single integer `n` will be expanded to `(n, n)`.
                     Powers of 2 are most efficient for FFT computation.
                     Default is the smallest power of 2 larger than the input, for each axis.

    Returns:
        Dict with keys

        -   `E_far`: Normalized E-field farfield; multiply by
                (i k exp(-i k r) / (4 pi r)) to get the actual field value.
        -   `H_far`: Normalized H-field farfield; multiply by
                (i k exp(-i k r) / (4 pi r)) to get the actual field value.
        -   `kx`, `ky`: Wavevector values corresponding to the x- and y- axes in E_far and H_far,
                normalized to wavelength (dimensionless).
        -   `dkx`, `dky`: step size for kx and ky, normalized to wavelength.
        -   `theta`: arctan2(ky, kx) corresponding to each (kx, ky).
                This is the angle in the x-y plane, counterclockwise from above, starting from +x.
        -   `phi`: arccos(kz / k) corresponding to each (kx, ky).
                This is the angle away from +z.
    """

    if not len(E_near) == 2:
        raise Exception('E_near must be a length-2 list of ndarrays')
    if not len(H_near) == 2:
        raise Exception('H_near must be a length-2 list of ndarrays')

    s = E_near[0].shape
    if not all(s == f.shape for f in E_near + H_near):
        raise Exception('All fields must be the same shape!')

    if padded_size is None:
        padded_size = (2**numpy.ceil(numpy.log2(s))).astype(int)
    if not hasattr(padded_size, '__len__'):
        padded_size = (padded_size, padded_size)            # type: ignore  # checked if sequence

    En_fft = [fftshift(fft2(fftshift(Eni), s=padded_size)) for Eni in E_near]
    Hn_fft = [fftshift(fft2(fftshift(Hni), s=padded_size)) for Hni in H_near]

    # Propagation vectors kx, ky
    k  = 2 * pi
    kxx = 2 * pi * fftshift(fftfreq(padded_size[0], dx))
    kyy = 2 * pi * fftshift(fftfreq(padded_size[1], dy))

    kx, ky = numpy.meshgrid(kxx, kyy, indexing='ij')
    kxy2 = kx * kx + ky * ky
    kxy = numpy.sqrt(kxy2)
    kz = numpy.sqrt(k * k - kxy2)

    sin_th = ky / kxy
    cos_th = kx / kxy
    cos_phi = kz / k

    sin_th[numpy.logical_and(kx == 0, ky == 0)] = 0
    cos_th[numpy.logical_and(kx == 0, ky == 0)] = 1

    # Normalized vector potentials N, L
    N = [-Hn_fft[1] * cos_phi * cos_th + Hn_fft[0] * cos_phi * sin_th,
          Hn_fft[1] * sin_th + Hn_fft[0] * cos_th]
    L = [ En_fft[1] * cos_phi * cos_th - En_fft[0] * cos_phi * sin_th,
         -En_fft[1] * sin_th - En_fft[0] * cos_th]

    E_far = [-L[1] - N[0],
              L[0] - N[1]]
    H_far = [-E_far[1],
              E_far[0]]

    theta = numpy.arctan2(ky, kx)
    phi = numpy.arccos(cos_phi)
    theta[numpy.logical_and(kx == 0, ky == 0)] = 0
    phi[numpy.logical_and(kx == 0, ky == 0)] = 0

    # Zero fields beyond valid (phi, theta)
    invalid_ind = kxy2 >= k * k
    theta[invalid_ind] = 0
    phi[invalid_ind] = 0
    for i in range(2):
        E_far[i][invalid_ind] = 0
        H_far[i][invalid_ind] = 0

    outputs = {
        'E': E_far,
        'H': H_far,
        'dkx': kx[1] - kx[0],
        'dky': ky[1] - ky[0],
        'kx': kx,
        'ky': ky,
        'theta': theta,
        'phi': phi,
    }

    return outputs


def far_to_nearfield(
        E_far: cfdfield_t,
        H_far: cfdfield_t,
        dkx: float,
        dky: float,
        padded_size: List[int] = None
        ) -> Dict[str, Any]:
    """
    Compute the farfield, i.e. the distribution of the fields after propagation
      through several wavelengths of uniform medium.

    The input fields should be complex phasors.

    Args:
        E_far: List of 2 ndarrays containing the 2D phasor field slices for the transverse
                E fields (e.g. [Ex, Ey] for calculating the nearfield toward the z-direction).
                Fields should be normalized so that
                E_far = E_far_actual / (i k exp(-i k r) / (4 pi r))
        H_far: List of 2 ndarrays containing the 2D phasor field slices for the transverse
                H fields (e.g. [Hx, hy] for calculating the nearfield toward the z-direction).
                Fields should be normalized so that
                H_far = H_far_actual / (i k exp(-i k r) / (4 pi r))
        dkx: kx discretization, in units of wavelength.
        dky: ky discretization, in units of wavelength.
        padded_size: Shape of the output. A single integer `n` will be expanded to `(n, n)`.
                     Powers of 2 are most efficient for FFT computation.
                     Default is the smallest power of 2 larger than the input, for each axis.

    Returns:
        Dict with keys

        -   `E`: E-field nearfield
        -   `H`: H-field nearfield
        -   `dx`, `dy`: spatial discretization, normalized to wavelength (dimensionless)
    """

    if not len(E_far) == 2:
        raise Exception('E_far must be a length-2 list of ndarrays')
    if not len(H_far) == 2:
        raise Exception('H_far must be a length-2 list of ndarrays')

    s = E_far[0].shape
    if not all(s == f.shape for f in E_far + H_far):
        raise Exception('All fields must be the same shape!')

    if padded_size is None:
        padded_size = (2 ** numpy.ceil(numpy.log2(s))).astype(int)
    if not hasattr(padded_size, '__len__'):
        padded_size = (padded_size, padded_size)            # type: ignore  # checked if sequence

    k = 2 * pi
    kxs = fftshift(fftfreq(s[0], 1 / (s[0] * dkx)))
    kys = fftshift(fftfreq(s[0], 1 / (s[1] * dky)))

    kx, ky = numpy.meshgrid(kxs, kys, indexing='ij')
    kxy2 = kx * kx + ky * ky
    kxy = numpy.sqrt(kxy2)

    kz = numpy.sqrt(k * k - kxy2)

    sin_th = ky / kxy
    cos_th = kx / kxy
    cos_phi = kz / k

    sin_th[numpy.logical_and(kx == 0, ky == 0)] = 0
    cos_th[numpy.logical_and(kx == 0, ky == 0)] = 1

    theta = numpy.arctan2(ky, kx)
    phi = numpy.arccos(cos_phi)
    theta[numpy.logical_and(kx == 0, ky == 0)] = 0
    phi[numpy.logical_and(kx == 0, ky == 0)] = 0

    # Zero fields beyond valid (phi, theta)
    invalid_ind = kxy2 >= k * k
    theta[invalid_ind] = 0
    phi[invalid_ind] = 0
    for i in range(2):
        E_far[i][invalid_ind] = 0
        H_far[i][invalid_ind] = 0

    # Normalized vector potentials N, L
    L = [0.5 * E_far[1],
        -0.5 * E_far[0]]
    N = [L[1],
        -L[0]]

    En_fft = [-( L[0] * sin_th + L[1] * cos_phi * cos_th) / cos_phi,
              -(-L[0] * cos_th + L[1] * cos_phi * sin_th) / cos_phi]

    Hn_fft = [( N[0] * sin_th + N[1] * cos_phi * cos_th) / cos_phi,
              (-N[0] * cos_th + N[1] * cos_phi * sin_th) / cos_phi]

    for i in range(2):
        En_fft[i][cos_phi == 0] = 0
        Hn_fft[i][cos_phi == 0] = 0

    E_near = [ifftshift(ifft2(ifftshift(Ei), s=padded_size)) for Ei in En_fft]
    H_near = [ifftshift(ifft2(ifftshift(Hi), s=padded_size)) for Hi in Hn_fft]

    dx = 2 * pi / (s[0] * dkx)
    dy = 2 * pi / (s[0] * dky)

    outputs = {
        'E': E_near,
        'H': H_near,
        'dx': dx,
        'dy': dy,
    }

    return outputs

