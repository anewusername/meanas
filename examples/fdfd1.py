import importlib
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
    grid = gridlock.Grid(edge_coords)
    epsilon = grid.allocate(n_air**2, dtype=numpy.float32)
    grid.draw_cuboid(epsilon, x=dict(center=0, span=8e3), y=dict(center=0, span=w), z=dict(center=0, span=th), foreground=n_wg**2)

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

    wg_results = waveguide_3d.solve_mode(mode_number=0, omega=omega, epsilon=epsilon, **wg_args)
    J = waveguide_3d.compute_source(E=wg_results['E'], wavenumber=wg_results['wavenumber'],
                                    omega=omega, epsilon=epsilon, **wg_args)
    e_overlap = waveguide_3d.compute_overlap_e(E=wg_results['E'], wavenumber=wg_results['wavenumber'], **wg_args)

    pecg = numpy.zeros_like(epsilon)
    # pecg.draw_cuboid(pecg, center=[700, 0, 0], dimensions=[80, 1e8, 1e8], eps=1)
    # pecg.visualize_isosurface(pecg)

    pmcg = numpy.zeros_like(epsilon)
    # grid.draw_cuboid(pmcg, center=[700, 0, 0], dimensions=[80, 1e8, 1e8], eps=1)
    # grid.visualize_isosurface(pmcg)

    grid.visualize_slice(J.imag, plane=dict(y=6*dx), which_shifts=1, pcolormesh_args=dict(norm=colors.CenteredNorm(), cmap='bwr'))
    fig, ax = pyplot.subplots()
    ax.pcolormesh((numpy.abs(J).sum(axis=2).sum(axis=0) > 0).astype(float).T, cmap='hot')
    pyplot.show(block=True)

    #
    # Solve!
    #
    sim_args = {
        'omega': omega,
        'dxes': dxes,
        'epsilon': vec(epsilon),
        'pec': vec(pecg),
        'pmc': vec(pmcg),
    }

    x = solver(J=vec(J), **sim_args)

    b = -1j * omega * vec(J)
    A = operators.e_full(**sim_args).tocsr()
    print('Norm of the residual is ', norm(A @ x - b))

    E = unvec(x, grid.shape)

    #
    # Plot results
    #
    center = grid.pos2ind([0, 0, 0], None).astype(int)
    fig, axes = pyplot.subplots(2, 2)
    grid.visualize_slice(E.real, plane=dict(x=0), which_shifts=1, ax=axes[0, 0], finalize=False, pcolormesh_args=dict(norm=colors.CenteredNorm(), cmap='bwr'))
    grid.visualize_slice(E.real, plane=dict(z=0), which_shifts=1, ax=axes[0, 1], finalize=False, pcolormesh_args=dict(norm=colors.CenteredNorm(), cmap='bwr'))
#    pcolor(axes[0, 0], numpy.real(E[1][center[0], :, :]).T)
#    pcolor(axes[0, 1], numpy.real(E[1][:, :, center[2]]).T)
    axes[1, 0].plot(numpy.log10(numpy.abs(E[1][:, center[1], center[2]]) + 1e-10))
    axes[1, 0].grid(alpha=0.6)
    axes[1, 0].set_ylabel('log10 of field')

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
    ax = axes[1, 1]
    ax.plot(s0x[0].sum(axis=2).sum(axis=1), label='s0', marker='.')
    ax.plot(s1x[0].sum(axis=2).sum(axis=1), label='s1', marker='.')
    ax.plot(s2x[0].sum(axis=2).sum(axis=1), label='s2', marker='.')
    ax.plot(E[1][:, center[1], center[2]].real.T, label='Ey', marker='x')
    ax.grid(alpha=0.6)
    ax.legend()

    p_in = (-E * J.conj()).sum() / 2 * (dx * dx * dx)
    print(f'{p_in=}')

    q = []
    for i in range(-5, 30):
        e_ovl_rolled = numpy.roll(e_overlap, i, axis=1)
        q += [numpy.abs(vec(E).conj() @ vec(e_ovl_rolled))]
    fig, ax = pyplot.subplots()
    ax.plot(q, marker='.')
    ax.grid(alpha=0.6)
    ax.set_title('Overlap with mode')
    print('Average overlap with mode:', sum(q[8:32])/len(q[8:32]))

    pyplot.show(block=True)


def module_available(name):
    return importlib.util.find_spec(name) is not None


if __name__ == '__main__':
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

