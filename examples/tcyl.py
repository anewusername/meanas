import importlib
import numpy
from numpy.linalg import norm

from meanas.fdmath import vec, unvec
from meanas.fdfd import waveguide_cyl, functional, scpml
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
    half_edge_coords = [numpy.arange(dx / 2, m + dx / 2, step=dx) for m in xyz_max]
    edge_coords = [numpy.hstack((-h[::-1], h)) for h in half_edge_coords]
    edge_coords[0] = numpy.array([-dx, dx])

    # #### Create the grid and draw the device ####
    grid = gridlock.Grid(edge_coords)
    epsilon = grid.allocate(n_air**2, dtype=numpy.float32)
    grid.draw_cuboid(epsilon, center=center, dimensions=[8e3, w, th], foreground=n_wg**2)

    dxes = [grid.dxyz, grid.autoshifted_dxyz()]
    for a in (1, 2):
        for p in (-1, 1):
            dxes = scpml.stretch_with_scpml(
                dxes,
                omega=omega,
                axis=a,
                polarity=p,
                thickness=pml_thickness,
                )

    wg_args = {
        'omega': omega,
        'dxes': [(d[1], d[2]) for d in dxes],
        'epsilon': vec(epsilon.transpose([0, 2, 3, 1])),
        'r0': r0,
    }

    wg_results = waveguide_cyl.solve_mode(mode_number=0, **wg_args)

    E = wg_results['E']

    n_eff = wl / (2 * numpy.pi / wg_results['wavenumber'])
    print('n =', n_eff)
    print('alpha (um^-1) =', -4 * numpy.pi * numpy.imag(n_eff) / (wl * 1e-3))

    '''
    Plot results
    '''
    def pcolor(fig, ax, v, title):
        vmax = numpy.max(numpy.abs(v))
        mappable = ax.pcolormesh(v.T, cmap='seismic', vmin=-vmax, vmax=vmax)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.figure.colorbar(mappable)

    fig, axes = pyplot.subplots(2, 2)
    pcolor(fig, axes[0][0], numpy.real(E[0]), 'Ex')
    pcolor(fig, axes[0][1], numpy.real(E[1]), 'Ey')
    pcolor(fig, axes[1][0], numpy.real(E[2]), 'Ez')
    pyplot.show()


if __name__ == '__main__':
    test1()
