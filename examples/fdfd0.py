import numpy
from numpy.linalg import norm
from matplotlib import pyplot, colors
import logging

import meanas
from meanas import fdtd
from meanas.fdmath import vec, unvec
from meanas.fdfd import waveguide_3d, functional, scpml, operators
from meanas.fdfd.solvers import generic as generic_solver

import gridlock


logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

__author__ = 'Jan Petykiewicz'


def pcolor(ax, v) -> None:
    mappable = ax.pcolor(v, cmap='seismic', norm=colors.CenteredNorm())
    ax.axis('equal')
    ax.get_figure().colorbar(mappable)


def test0(solver=generic_solver):
    dx = 50                # discretization (nm/cell)
    pml_thickness = 10     # (number of cells)

    wl = 1550               # Excitation wavelength
    omega = 2 * numpy.pi / wl

    # Device design parameters
    radii = (1, 0.6)
    th = 220
    center = [0, 0, 0]

    # refractive indices
    n_ring = numpy.sqrt(12.6)  # ~Si
    n_air = 4.0   # air

    # Half-dimensions of the simulation grid
    xyz_max = numpy.array([1.2, 1.2, 0.3]) * 1000 + pml_thickness * dx

    # Coordinates of the edges of the cells.
    half_edge_coords = [numpy.arange(dx/2, m + dx, step=dx) for m in xyz_max]
    edge_coords = [numpy.hstack((-h[::-1], h)) for h in half_edge_coords]

    # #### Create the grid, mask, and draw the device ####
    grid = gridlock.Grid(edge_coords)
    epsilon = grid.allocate(n_air**2, dtype=numpy.float32)
    grid.draw_cylinder(
        epsilon,
        h = dict(axis='z', center=center[2], span=th),
        radius = max(radii),
        center2d = center[:2],
        foreground = n_ring ** 2,
        num_points = 24,
        )
    grid.draw_cylinder(
        epsilon,
        h = dict(axis='z', center=center[2], span=th * 1.1),
        radius = min(radii),
        center2d = center[:2],
        foreground = n_air ** 2,
        num_points = 24,
        )

    dxes = [grid.dxyz, grid.autoshifted_dxyz()]
    for a in (0, 1, 2):
        for p in (-1, 1):
            dxes = meanas.fdfd.scpml.stretch_with_scpml(dxes, axis=a, polarity=p, omega=omega,
                                                        thickness=pml_thickness)

    J = [numpy.zeros_like(epsilon[0], dtype=complex) for _ in range(3)]
    J[1][15, grid.shape[1]//2, grid.shape[2]//2] = 1


    #
    # Solve!
    #
    sim_args = dict(
        omega = omega,
        dxes = dxes,
        epsilon = vec(epsilon),
        )
    x = solver(J=vec(J), **sim_args)

    A = operators.e_full(omega, dxes, vec(epsilon)).tocsr()
    b = -1j * omega * vec(J)
    print('Norm of the residual is ', norm(A @ x - b) / norm(b))

    E = unvec(x, grid.shape)

    #
    # Plot results
    #
    grid.visualize_slice(E.real, plane=dict(z=0), which_shifts=1, pcolormesh_args=dict(norm=colors.CenteredNorm(), cmap='bwr'))


if __name__ == '__main__':
    test0()
