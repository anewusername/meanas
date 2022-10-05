"""
Functions for creating stretched coordinate perfectly matched layer (PML) absorbers.
"""

from typing import Sequence, Union, Callable, Optional, List

import numpy
from numpy.typing import ArrayLike, NDArray


__author__ = 'Jan Petykiewicz'


s_function_t = Callable[[NDArray[numpy.float64]], NDArray[numpy.float64]]
"""Typedef for s-functions, see `prepare_s_function()`"""


def prepare_s_function(
        ln_R: float = -16,
        m: float = 4
        ) -> s_function_t:
    """
    Create an s_function to pass to the SCPML functions. This is used when you would like to
    customize the PML parameters.

    Args:
        ln_R: Natural logarithm of the desired reflectance
        m: Polynomial order for the PML (imaginary part increases as distance ** m)

    Returns:
        An s_function, which takes an ndarray (distances) and returns an ndarray (complex part
        of the cell width; needs to be divided by `sqrt(epilon_effective) * real(omega))`
        before use.
    """
    def s_factor(distance: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        s_max = (m + 1) * ln_R / 2  # / 2 because we assume periodic boundaries
        return s_max * (distance ** m)
    return s_factor


def uniform_grid_scpml(
        shape: Sequence[int],
        thicknesses: Sequence[int],
        omega: float,
        epsilon_effective: float = 1.0,
        s_function: Optional[s_function_t] = None,
        ) -> List[List[NDArray[numpy.float64]]]:
    """
    Create dx arrays for a uniform grid with a cell width of 1 and a pml.

    If you want something more fine-grained, check out `stretch_with_scpml(...)`.

    Args:
        shape: Shape of the grid, including the PMLs (which are 2*thicknesses thick)
        thicknesses: `[th_x, th_y, th_z]`
                     Thickness of the PML in each direction.
                     Both polarities are added.
                     Each th_ of pml is applied twice, once on each edge of the grid along the given axis.
                     `th_*` may be zero, in which case no pml is added.
        omega: Angular frequency for the simulation
        epsilon_effective: Effective epsilon of the PML. Match this to the material
                            at the edge of your grid.
                            Default 1.
        s_function: created by `prepare_s_function(...)`, allowing customization of pml parameters.
                    Default uses `prepare_s_function()` with no parameters.

    Returns:
        Complex cell widths (dx_lists_mut) as discussed in `meanas.fdmath.types`.
    """
    if s_function is None:
        s_function = prepare_s_function()

    shape = tuple(shape)
    thicknesses = tuple(thicknesses)

    # Normalized distance to nearest boundary
    def ll(u: NDArray[numpy.float64], n: int, t: int) -> NDArray[numpy.float64]:
        return ((t - u).clip(0) + (u - (n - t)).clip(0)) / t

    dx_a = [numpy.array(numpy.inf)] * 3
    dx_b = [numpy.array(numpy.inf)] * 3

    # divide by this to adjust for epsilon_effective and omega
    s_correction = numpy.sqrt(epsilon_effective) * numpy.real(omega)

    for k, th in enumerate(thicknesses):
        s = shape[k]
        if th > 0:
            sr = numpy.arange(s)
            dx_a[k] = 1 + 1j * s_function(ll(sr,       s, th)) / s_correction
            dx_b[k] = 1 + 1j * s_function(ll(sr + 0.5, s, th)) / s_correction
        else:
            dx_a[k] = numpy.ones((s,))
            dx_b[k] = numpy.ones((s,))
    return [dx_a, dx_b]


def stretch_with_scpml(
        dxes: List[List[NDArray[numpy.float64]]],
        axis: int,
        polarity: int,
        omega: float,
        epsilon_effective: float = 1.0,
        thickness: int = 10,
        s_function: Optional[s_function_t] = None,
        ) -> List[List[NDArray[numpy.float64]]]:
    """
        Stretch dxes to contain a stretched-coordinate PML (SCPML) in one direction along one axis.

        Args:
            dxes: Grid parameters `[dx_e, dx_h]` as described in `meanas.fdmath.types`
            axis: axis to stretch (0=x, 1=y, 2=z)
            polarity: direction to stretch (-1 for -ve, +1 for +ve)
            omega: Angular frequency for the simulation
            epsilon_effective: Effective epsilon of the PML. Match this to the material at the
                               edge of your grid. Default 1.
            thickness: number of cells to use for pml (default 10)
            s_function: Created by `prepare_s_function(...)`, allowing customization
                        of pml parameters. Default uses `prepare_s_function()` with no parameters.

        Returns:
            Complex cell widths (dx_lists_mut) as discussed in `meanas.fdmath.types`.
            Multiple calls to this function may be necessary if multiple absorpbing boundaries are needed.
    """
    if s_function is None:
        s_function = prepare_s_function()

    dx_ai = dxes[0][axis].astype(complex)
    dx_bi = dxes[1][axis].astype(complex)

    pos = numpy.hstack((0, dx_ai.cumsum()))
    pos_a = (pos[:-1] + pos[1:]) / 2
    pos_b = pos[:-1]

    # divide by this to adjust for epsilon_effective and omega
    s_correction = numpy.sqrt(epsilon_effective) * numpy.real(omega)

    if polarity > 0:
        # front pml
        bound = pos[thickness]
        d = bound - pos[0]

        def l_d(x: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
            return (bound - x) / (bound - pos[0])

        slc = slice(thickness)

    else:
        # back pml
        bound = pos[-thickness - 1]
        d = pos[-1] - bound

        def l_d(x: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
            return (x - bound) / (pos[-1] - bound)

        if thickness == 0:
            slc = slice(None)
        else:
            slc = slice(-thickness, None)

    dx_ai[slc] *= 1 + 1j * s_function(l_d(pos_a[slc])) / d / s_correction
    dx_bi[slc] *= 1 + 1j * s_function(l_d(pos_b[slc])) / d / s_correction

    dxes[0][axis] = dx_ai
    dxes[1][axis] = dx_bi

    return dxes
