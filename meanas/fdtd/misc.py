from collections.abc import Callable
import logging

import numpy
from numpy.typing import NDArray, ArrayLike
from numpy import pi


logger = logging.getLogger(__name__)


pulse_fn_t = Callable[[int | NDArray], tuple[float, float, float]]


def gaussian_packet(
        wl: float,
        dwl: float,
        dt: float,
        turn_on: float = 1e-10,
        one_sided: bool = False,
        ) -> tuple[pulse_fn_t, float]:
    """
    Gaussian pulse (or gaussian ramp) for FDTD excitation

    exp(-a*t*t) ==> exp(-omega * omega / (4 * a))   [fourier, ignoring leading const.]

    FWHM_time is 2 * sqrt(2 * log(2)) * sqrt(2 / a)
    FWHM_omega is 2 * sqrt(2 * log(2)) * sqrt(2 * a)  = 4 * sqrt(log(2) * a)

    Args:
        wl: wavelength
        dwl: Gaussian's FWHM in wavelength space
        dt: Timestep
        turn_on: Max allowable amplitude at t=0
        one_sided: If `True`, source amplitude never decreases after reaching max

    Returns:
        Source function: src(timestep) -> (envelope[tt], cos[... * tt], sin[... * tt])
        Delay: number of initial timesteps for which envelope[tt] will be 0
    """
    logger.warning('meanas.fdtd.misc functions are still very WIP!')    # TODO
    # dt * dw = 4 * ln(2)

    omega = 2 * pi / wl
    freq = 1 / wl
    fwhm_omega = dwl * omega * omega / (2 * pi)         # dwl -> d_omega (approx)
    alpha = (fwhm_omega * fwhm_omega) * numpy.log(2) / 8
    delay = numpy.sqrt(-numpy.log(turn_on) / alpha)
    delay = numpy.ceil(delay * freq) / freq     # force delay to integer number of periods to maintain phase
    logger.info(f'src_time {2 * delay / dt}')

    def source_phasor(ii: int | NDArray) -> tuple[float, float, float]:
        t0 = ii * dt - delay
        envelope = numpy.sqrt(numpy.sqrt(2 * alpha / pi)) * numpy.exp(-alpha * t0 * t0)

        if one_sided and t0 > 0:
            envelope = 1

        cc = numpy.cos(omega * t0)
        ss = numpy.sin(omega * t0)

        return envelope, cc, ss

    # nrm = numpy.exp(-omega * omega / alpha) / 2

    return source_phasor, delay


def ricker_pulse(
        wl: float,
        dt: float,
        turn_on: float = 1e-10,
        ) -> tuple[pulse_fn_t, float]:
    """
    Ricker wavelet (second derivative of a gaussian pulse)

    t0 = ii * dt - delay
    R = w_peak * t0 / 2
    f(t) = (1 - 2 * (pi * f_peak * t0) ** 2) * exp(-(pi * f_peak * t0)**2
         = (1 - (w_peak * t0)**2 / 2 exp(-(w_peak * t0 / 2) **2)
         = (1 - 2 * R * R) * exp(-R * R)

    # NOTE: don't use cosine/sine for J, just for phasor readout

    Args:
        wl: wavelength
        dt: Timestep
        turn_on: Max allowable amplitude at t=0

    Returns:
        Source function: src(timestep) -> (envelope[tt], cos[... * tt], sin[... * tt])
        Delay: number of initial timesteps for which envelope[tt] will be 0
    """
    logger.warning('meanas.fdtd.misc functions are still very WIP!')    # TODO
    omega = 2 * pi / wl
    freq = 1 / wl
    # r0 = omega / 2

    from scipy.optimize import root_scalar
    delay_results = root_scalar(lambda tt: (1 - omega * omega * tt * tt / 2) * numpy.exp(-omega * omega / 4 * tt * tt) - turn_on, x0=0, x1=-2 / omega)
    delay = delay_results.root
    delay = numpy.ceil(delay * freq) / freq     # force delay to integer number of periods to maintain phase

    def source_phasor(ii: int | NDArray) -> tuple[float, float, float]:
        t0 = ii * dt - delay
        rr = omega * t0 / 2
        ff = (1 - 2 * rr * rr) * numpy.exp(-rr * rr)

        cc = numpy.cos(omega * t0)
        ss = numpy.sin(omega * t0)

        return ff, cc, ss

    return source_phasor, delay


def gaussian_beam(
        xyz: list[NDArray],
        center: ArrayLike,
        waist_radius: float,
        wl: float,
        tilt: float = 0,
        ) -> NDArray[numpy.complex128]:
    """
    Gaussian beam
    (solution to paraxial Helmholtz equation)

    Default (no tilt) corresponds to a beam propagating in the -z direction.

    Args:
        xyz: List of [[x0, x1, ...], [y0, ...], [z0, ...]] positions specifying grid
            locations at which the field will be sampled.
        center: [x, y, z] location of beam waist
        waist_radius: Beam radius at the waist
        wl: Wavelength
        tilt: Rotation around y axis. Default (0) has beam propagating in -z direction.
    """
    logger.warning('meanas.fdtd.misc functions are still very WIP!')    # TODO
    w0 = waist_radius
    grids = numpy.asarray(numpy.meshgrid(*xyz, indexing='ij'))
    grids -= numpy.asarray(center)[:, None, None, None]

    rot = numpy.array([
        [ numpy.cos(tilt), 0, numpy.sin(tilt)],
        [               0, 1,               0],
        [-numpy.sin(tilt), 0, numpy.cos(tilt)],
        ])

    xx, yy, zz = numpy.einsum('ij,jxyz->ixyz', rot, grids)
    r2 = xx * xx + yy * yy
    z2 = zz * zz

    zr = pi * w0 * w0 / wl
    zr2 = zr * zr
    wz2 = w0 * w0 * (1 + z2 / zr2)
    wz = numpy.sqrt(wz2)        # == fwhm(z) / sqrt(2 * ln(2))

    kk = 2 * pi / wl
    Rz = zz * (1 + zr2 / z2)
    gouy = numpy.arctan(zz / zr)

    gaussian = w0 / wz * numpy.exp(-r2 / wz2) * numpy.exp(1j * (kk * zz + kk * r2 / 2 / Rz - gouy))

    row = gaussian[:, :, gaussian.shape[2] // 2]
    norm = numpy.sqrt((row * row.conj()).sum())
    return gaussian / norm
