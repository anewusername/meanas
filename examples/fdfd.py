import importlib
import numpy
from numpy.linalg import norm

from fdfd_tools import vec, unvec, waveguide_mode
import fdfd_tools
import fdfd_tools.functional
import fdfd_tools.grid
from fdfd_tools.solvers import generic as generic_solver

import gridlock

from matplotlib import pyplot


__author__ = 'Jan Petykiewicz'


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
    grid = gridlock.Grid(edge_coords, initial=n_air**2, num_grids=3)
    grid.draw_cylinder(surface_normal=gridlock.Direction.z,
                       center=center,
                       radius=max(radii),
                       thickness=th,
                       eps=n_ring**2,
                       num_points=24)
    grid.draw_cylinder(surface_normal=gridlock.Direction.z,
                       center=center,
                       radius=min(radii),
                       thickness=th*1.1,
                       eps=n_air ** 2,
                       num_points=24)

    dxes = [grid.dxyz, grid.autoshifted_dxyz()]
    for a in (0, 1, 2):
        for p in (-1, 1):
            dxes = fdfd_tools.grid.stretch_with_scpml(dxes, axis=a, polarity=p, omega=omega,
                                                      thickness=pml_thickness)

    J = [numpy.zeros_like(grid.grids[0], dtype=complex) for _ in range(3)]
    J[1][15, grid.shape[1]//2, grid.shape[2]//2] = 1e5

    '''
    Solve!
    '''
    x = solver(J=vec(J), **sim_args)

    A = fdfd_tools.functional.e_full(omega, dxes, vec(grid.grids)).tocsr()
    b = -1j * omega * vec(J)
    print('Norm of the residual is ', norm(A @ x - b))

    E = unvec(x, grid.shape)

    '''
    Plot results
    '''
    pyplot.figure()
    pyplot.pcolor(numpy.real(E[1][:, :, grid.shape[2]//2]), cmap='seismic')
    pyplot.axis('equal')
    pyplot.show()


def test1(solver=generic_solver):
    dx = 40                 # discretization (nm/cell)
    pml_thickness = 10      # (number of cells)

    wl = 1550               # Excitation wavelength
    omega = 2 * numpy.pi / wl

    # Device design parameters
    w = 600
    th = 220
    center = [0, 0, 0]

    # refractive indices
    n_wg = numpy.sqrt(12.6)  # ~Si
    n_air = 1.0              # air

    # Half-dimensions of the simulation grid
    xyz_max = numpy.array([0.8, 0.9, 0.6]) * 1000 + (pml_thickness + 2) * dx

    # Coordinates of the edges of the cells.
    half_edge_coords = [numpy.arange(dx/2, m + dx/2, step=dx) for m in xyz_max]
    edge_coords = [numpy.hstack((-h[::-1], h)) for h in half_edge_coords]

    # #### Create the grid and draw the device ####
    grid = gridlock.Grid(edge_coords, initial=n_air**2, num_grids=3)
    grid.draw_cuboid(center=center, dimensions=[8e3, w, th], eps=n_wg**2)

    dxes = [grid.dxyz, grid.autoshifted_dxyz()]
    for a in (0, 1, 2):
        for p in (-1, 1):
            dxes = fdfd_tools.grid.stretch_with_scpml(dxes,omega=omega, axis=a, polarity=p,
                                                      thickness=pml_thickness)

    half_dims = numpy.array([10, 20, 15]) * dx
    dims = [-half_dims, half_dims]
    dims[1][0] = dims[0][0]
    ind_dims = (grid.pos2ind(dims[0], which_shifts=None).astype(int),
                grid.pos2ind(dims[1], which_shifts=None).astype(int))
    wg_args = {
        'omega': omega,
        'slices': [slice(i, f+1) for i, f in zip(*ind_dims)],
        'dxes': dxes,
        'axis': 0,
        'polarity': +1,
    }

    wg_results = waveguide_mode.solve_waveguide_mode(mode_number=0, **wg_args, epsilon=grid.grids)
    J = waveguide_mode.compute_source(**wg_args, **wg_results)
    H_overlap = waveguide_mode.compute_overlap_e(**wg_args, **wg_results)

    pecg = gridlock.Grid(edge_coords, initial=0.0, num_grids=3)
    # pecg.draw_cuboid(center=[700, 0, 0], dimensions=[80, 1e8, 1e8], eps=1)
    # pecg.visualize_isosurface()

    pmcg = gridlock.Grid(edge_coords, initial=0.0, num_grids=3)
    # pmcg.draw_cuboid(center=[700, 0, 0], dimensions=[80, 1e8, 1e8], eps=1)
    # pmcg.visualize_isosurface()

    '''
    Solve!
    '''
    sim_args = {
        'omega': omega,
        'dxes': dxes,
        'epsilon': vec(grid.grids),
        'pec': vec(pecg.grids),
        'pmc': vec(pmcg.grids),
    }

    x = solver(J=vec(J), **sim_args)

    b = -1j * omega * vec(J)
    A = fdfd_tools.operators.e_full(**sim_args).tocsr()
    print('Norm of the residual is ', norm(A @ x - b))

    E = unvec(x, grid.shape)

    '''
    Plot results
    '''
    def pcolor(v):
        vmax = numpy.max(numpy.abs(v))
        pyplot.pcolor(v, cmap='seismic', vmin=-vmax, vmax=vmax)
        pyplot.axis('equal')
        pyplot.colorbar()

    center = grid.pos2ind([0, 0, 0], None).astype(int)
    pyplot.figure()
    pyplot.subplot(2, 2, 1)
    pcolor(numpy.real(E[1][center[0], :, :]))
    pyplot.subplot(2, 2, 2)
    pyplot.plot(numpy.log10(numpy.abs(E[1][:, center[1], center[2]]) + 1e-10))
    pyplot.subplot(2, 2, 3)
    pcolor(numpy.real(E[1][:, :, center[2]]))
    pyplot.subplot(2, 2, 4)

    def poyntings(E):
        e = vec(E)
        h = fdfd_tools.operators.e2h(omega, dxes) @ e
        cross1 = fdfd_tools.operators.poynting_e_cross(e, dxes) @ h.conj()
        cross2 = fdfd_tools.operators.poynting_h_cross(h.conj(), dxes) @ e
        s1 = unvec(0.5 * numpy.real(cross1), grid.shape)
        s2 = unvec(0.5 * numpy.real(-cross2), grid.shape)
        return s1, s2

    s1x, s2x = poyntings(E)
    pyplot.plot(s1x[0].sum(axis=2).sum(axis=1))
    pyplot.plot(s2x[0].sum(axis=2).sum(axis=1))
    pyplot.show()

    q = []
    for i in range(-5, 30):
        H_rolled = [numpy.roll(h, i, axis=0) for h in H_overlap]
        q += [numpy.abs(vec(E) @ vec(H_rolled))]
    pyplot.figure()
    pyplot.plot(q)
    pyplot.title('Overlap with mode')
    pyplot.show()
    print('Average overlap with mode:', sum(q)/len(q))


def module_available(name):
    return importlib.util.find_spec(name) is not None


if __name__ == '__main__':
    # test0()

    if module_available('opencl_fdfd'):
        from opencl_fdfd import cg_solver as opencl_solver
        test1(opencl_solver)
        # from opencl_fdfd.csr import fdfd_cg_solver as opencl_csr_solver
        # test1(opencl_csr_solver)
    # elif module_available('magma_fdfd'):
    #     from magma_fdfd import solver as magma_solver
    #     test1(magma_solver)
    else:
        test1()