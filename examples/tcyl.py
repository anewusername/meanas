import importlib
import numpy
from numpy.linalg import norm

from meanas.fdmath import vec, unvec
from meanas.fdfd import waveguide_mode, functional, scpml
from meanas.fdfd.solvers import generic as generic_solver

import gridlock

from matplotlib import pyplot


__author__ = 'Jan Petykiewicz'


def test1(solver=generic_solver):
    dx = 20                 # discretization (nm/cell)
    pml_thickness = 10      # (number of cells)

    wl = 1550               # Excitation wavelength
    omega = 2 * numpy.pi / wl

    # Device design parameters
    w = 800
    th = 220
    center = [0, 0, 0]
    r0 = 8e3

    # refractive indices
    n_wg = numpy.sqrt(12.6)  # ~Si
    n_air = 1.0              # air

    # Half-dimensions of the simulation grid
    y_max = 1200
    z_max = 900
    xyz_max = numpy.array([800, y_max, z_max]) + (pml_thickness + 2) * dx

    # Coordinates of the edges of the cells.
    half_edge_coords = [numpy.arange(dx/2, m + dx/2, step=dx) for m in xyz_max]
    edge_coords = [numpy.hstack((-h[::-1], h)) for h in half_edge_coords]
    edge_coords[0] = numpy.array([-dx, dx])

    # #### Create the grid and draw the device ####
    grid = gridlock.Grid(edge_coords)
    epsilon = grid.allocate(n_air**2, dtype=numpy.float32)
    grid.draw_cuboid(epsilon, center=center, dimensions=[8e3, w, th], eps=n_wg**2)

    dxes = [grid.dxyz, grid.autoshifted_dxyz()]
    for a in (1, 2):
        for p in (-1, 1):
            dxes = scmpl.stretch_with_scpml(dxes, omega=omega, axis=a, polarity=p,
                                            thickness=pml_thickness)

    wg_args = {
        'omega': omega,
        'dxes': [(d[1], d[2]) for d in dxes],
        'epsilon': vec(g.transpose([1, 2, 0]) for g in epsilon),
        'r0': r0,
    }

    wg_results = waveguide_mode.solve_waveguide_mode_cylindrical(mode_number=0, **wg_args)

    E = wg_results['E']

    n_eff = wl / (2 * numpy.pi / wg_results['wavenumber'])
    print('n =', n_eff)
    print('alpha (um^-1) =', -4 * numpy.pi * numpy.imag(n_eff) / (wl * 1e-3))

    '''
    Plot results
    '''
    def pcolor(v):
        vmax = numpy.max(numpy.abs(v))
        pyplot.pcolor(v.T, cmap='seismic', vmin=-vmax, vmax=vmax)
        pyplot.axis('equal')
        pyplot.colorbar()

    pyplot.figure()
    pyplot.subplot(2, 2, 1)
    pcolor(numpy.real(E[0][:, :]))
    pyplot.subplot(2, 2, 2)
    pcolor(numpy.real(E[1][:, :]))
    pyplot.subplot(2, 2, 3)
    pcolor(numpy.real(E[2][:, :]))
    pyplot.subplot(2, 2, 4)
    pyplot.show()


if __name__ == '__main__':
    test1()
