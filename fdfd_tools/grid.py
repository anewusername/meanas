"""
Functions for creating stretched coordinate PMLs.
"""

from typing import List, Tuple, Callable
import numpy

__author__ = 'Jan Petykiewicz'


dx_lists_t = List[List[numpy.ndarray]]
s_function_type = Callable[[float], float]


def prepare_s_function(ln_R: float = -16,
                       m: float = 4
                       ) -> s_function_type:
    """
    Create an s_function to pass to the SCPML functions. This is used when you would like to
    customize the PML parameters.

    :param ln_R: Natural logarithm of the desired reflectance
    :param m: Polynomial order for the PML (imaginary part increases as distance ** m)
    :return: An s_function, which takes an ndarray (distances) and returns an ndarray (complex part of
                the cell width; needs to be divided by sqrt(epilon_effective) * real(omega)) before
                use.
    """
    def s_factor(distance: numpy.ndarray) -> numpy.ndarray:
        s_max = (m + 1) * ln_R / 2  # / 2 because we assume periodic boundaries
        return s_max * (distance ** m)
    return s_factor


def uniform_grid_scpml(shape: numpy.ndarray or List[int],
                       thicknesses: numpy.ndarray or List[int],
                       omega: float,
                       epsilon_effective: float = 1.0,
                       s_function: s_function_type = None,
                       ) -> dx_lists_t:
    """
    Create dx arrays for a uniform grid with a cell width of 1 and a pml.

    If you want something more fine-grained, check out stretch_with_scpml(...).

    :param shape: Shape of the grid, including the PMLs (which are 2*thicknesses thick)
    :param thicknesses: [th_x, th_y, th_z] Thickness of the PML in each direction.
        Both polarities are added.
        Each th_ of pml is applied twice, once on each edge of the grid along the given axis.
        th_* may be zero, in which case no pml is added.
    :param omega: Angular frequency for the simulation
    :param epsilon_effective: Effective epsilon of the PML. Match this to the material
        at the edge of your grid.
        Default 1.
    :param s_function: s_function created by prepare_s_function(...), allowing
        customization of pml parameters.
        Default uses prepare_s_function() with no parameters.
    :return: Complex cell widths (dx_lists)
    """
    if s_function is None:
        s_function = prepare_s_function()

    # Normalized distance to nearest boundary
    def l(u, n, t):
        return ((t - u).clip(0) + (u - (n - t)).clip(0)) / t

    dx_a = [numpy.array(numpy.inf)] * 3
    dx_b = [numpy.array(numpy.inf)] * 3

    # divide by this to adjust for epsilon_effective and omega
    s_correction = numpy.sqrt(epsilon_effective) * numpy.real(omega)

    for k, th in enumerate(thicknesses):
        s = shape[k]
        if th > 0:
            sr = numpy.arange(s)
            dx_a[k] = 1 + 1j * s_function(l(sr, s, th)) / s_correction
            dx_b[k] = 1 + 1j * s_function(l(sr+0.5, s, th)) / s_correction
        else:
            dx_a[k] = numpy.ones((s,))
            dx_b[k] = numpy.ones((s,))
    return [dx_a, dx_b]


def stretch_with_scpml(dxes: dx_lists_t,
                       axis: int,
                       polarity: int,
                       omega: float,
                       epsilon_effective: float = 1.0,
                       thickness: int = 10,
                       s_function: s_function_type = None,
                       ) -> dx_lists_t:
    """
        Stretch dxes to contain a stretched-coordinate PML (SCPML) in one direction along one axis.

        :param dxes: dx_tuple with coordinates to stretch
        :param axis: axis to stretch (0=x, 1=y, 2=z)
        :param polarity: direction to stretch (-1 for -ve, +1 for +ve)
        :param omega: Angular frequency for the simulation
        :param epsilon_effective: Effective epsilon of the PML. Match this to the material at the edge of your grid.
            Default 1.
        :param thickness: number of cells to use for pml (default 10)
        :param s_function: s_function created by prepare_s_function(...), allowing customization of pml parameters.
            Default uses prepare_s_function() with no parameters.
        :return: Complex cell widths
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

        def l_d(x):
            return (bound - x) / (bound - pos[0])

        slc = slice(thickness)

    else:
        # back pml
        bound = pos[-thickness - 1]
        d = pos[-1] - bound

        def l_d(x):
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


def generate_periodic_dx(pos: List[numpy.ndarray]) -> DXList:
    """
    Given a list of 3 ndarrays cell centers, creates the cell width parameters for a periodic grid.

    :param pos: List of 3 ndarrays of cell centers
    :return: (dx_a, dx_b) cell widths (no pml)
    """
    if len(pos) != 3:
        raise Exception('Must have len(pos) == 3')

    dx_a = [numpy.array(numpy.inf)] * 3
    dx_b = [numpy.array(numpy.inf)] * 3

    for i, p_orig in enumerate(pos):
        p = numpy.array(p_orig, dtype=float)
        if p.size != 1:
            p_shifted = numpy.hstack((p[1:], p[-1] + (p[1] - p[0])))
            dx_a[i] = numpy.diff(p)
            dx_b[i] = numpy.diff((p + p_shifted) / 2)
    return dx_a, dx_b
