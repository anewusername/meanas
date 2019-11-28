import importlib
import numpy
from numpy.linalg import norm

import meanas
from meanas import fdtd
from meanas.fdmath import vec, unvec
from meanas.fdfd import waveguide_3d, functional, scpml, operators
from meanas.fdfd.solvers import generic as generic_solver

import gridlock

from matplotlib import pyplot

import logging

logging.basicConfig(level=logging.DEBUG)

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
            dxes = meanas.fdfd.scpml.stretch_with_scpml(dxes, axis=a, polarity=p, omega=omega,
                                                        thickness=pml_thickness)

    J = [numpy.zeros_like(grid.grids[0], dtype=complex) for _ in range(3)]
    J[1][15, grid.shape[1]//2, grid.shape[2]//2] = 1


    '''
    Solve!
    '''
    sim_args = {
        'omega': omega,
        'dxes': dxes,
        'epsilon': vec(grid.grids),
    }
    x = solver(J=vec(J), **sim_args)

    A = operators.e_full(omega, dxes, vec(grid.grids)).tocsr()
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
            dxes = scpml.stretch_with_scpml(dxes,omega=omega, axis=a, polarity=p,
                                            thickness=pml_thickness)

    half_dims = numpy.array([10, 20, 15]) * dx
    dims = [-half_dims, half_dims]
    dims[1][0] = dims[0][0]
    ind_dims = (grid.pos2ind(dims[0], which_shifts=None).astype(int),
                grid.pos2ind(dims[1], which_shifts=None).astype(int))
    src_axis = 0
    wg_args = {
        'slices': [slice(i, f+1) for i, f in zip(*ind_dims)],
        'dxes': dxes,
        'axis': src_axis,
        'polarity': +1,
    }

    wg_results = waveguide_3d.solve_mode(mode_number=0, omega=omega, epsilon=grid.grids, **wg_args)
    J = waveguide_3d.compute_source(E=wg_results['E'], wavenumber=wg_results['wavenumber'],
                                    omega=omega, epsilon=grid.grids, **wg_args)
    e_overlap = waveguide_3d.compute_overlap_e(E=wg_results['E'], wavenumber=wg_results['wavenumber'], **wg_args)

    pecg = gridlock.Grid(edge_coords, initial=0.0, num_grids=3)
    # pecg.draw_cuboid(center=[700, 0, 0], dimensions=[80, 1e8, 1e8], eps=1)
    # pecg.visualize_isosurface()

    pmcg = gridlock.Grid(edge_coords, initial=0.0, num_grids=3)
    # pmcg.draw_cuboid(center=[700, 0, 0], dimensions=[80, 1e8, 1e8], eps=1)
    # pmcg.visualize_isosurface()

    def pcolor(v):
        vmax = numpy.max(numpy.abs(v))
        pyplot.pcolor(v, cmap='seismic', vmin=-vmax, vmax=vmax)
        pyplot.axis('equal')
        pyplot.colorbar()

    ss = (1, slice(None), J.shape[2]//2+6, slice(None))
#    pyplot.figure()
#    pcolor(J3[ss].T.imag)
#    pyplot.figure()
#    pcolor((numpy.abs(J3).sum(axis=2).sum(axis=0) > 0).astype(float).T)
    pyplot.show(block=True)

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
    A = operators.e_full(**sim_args).tocsr()
    print('Norm of the residual is ', norm(A @ x - b))

    E = unvec(x, grid.shape)

    '''
    Plot results
    '''
    center = grid.pos2ind([0, 0, 0], None).astype(int)
    pyplot.figure()
    pyplot.subplot(2, 2, 1)
    pcolor(numpy.real(E[1][center[0], :, :]).T)
    pyplot.subplot(2, 2, 2)
    pyplot.plot(numpy.log10(numpy.abs(E[1][:, center[1], center[2]]) + 1e-10))
    pyplot.grid(alpha=0.6)
    pyplot.ylabel('log10 of field')
    pyplot.subplot(2, 2, 3)
    pcolor(numpy.real(E[1][:, :, center[2]]).T)
    pyplot.subplot(2, 2, 4)

    def poyntings(E):
        H = functional.e2h(omega, dxes)(E)
        poynting = fdtd.poynting(e=E, h=H.conj(), dxes=dxes)
        cross1 = operators.poynting_e_cross(vec(E), dxes) @ vec(H).conj()
        cross2 = operators.poynting_h_cross(vec(H), dxes) @ vec(E).conj() * -1
        s1 = 0.5 * unvec(numpy.real(cross1), grid.shape)
        s2 = 0.5 * unvec(numpy.real(cross2), grid.shape)
        s0 = 0.5 * poynting.real
#        s2 = poynting.imag
        return s0, s1, s2

    s0x, s1x, s2x = poyntings(E)
    pyplot.plot(s0x[0].sum(axis=2).sum(axis=1), label='s0', marker='.')
    pyplot.plot(s1x[0].sum(axis=2).sum(axis=1), label='s1', marker='.')
    pyplot.plot(s2x[0].sum(axis=2).sum(axis=1), label='s2', marker='.')
    pyplot.plot(E[1][:, center[1], center[2]].real.T, label='Ey', marker='x')
    pyplot.grid(alpha=0.6)
    pyplot.legend()
    pyplot.show()

    q = []
    for i in range(-5, 30):
        e_ovl_rolled = numpy.roll(e_overlap, i, axis=1)
        q += [numpy.abs(vec(E) @ vec(e_ovl_rolled).conj())]
    pyplot.figure()
    pyplot.plot(q, marker='.')
    pyplot.grid(alpha=0.6)
    pyplot.title('Overlap with mode')
    pyplot.show()
    print('Average overlap with mode:', sum(q)/len(q))


def module_available(name):
    return importlib.util.find_spec(name) is not None


if __name__ == '__main__':
    #test0()
#    test1()

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
