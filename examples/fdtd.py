"""
Example code for running an OpenCL FDTD simulation

See main() for simulation setup.
"""

import sys
import time

import numpy
import h5py

from meanas import fdtd
from meanas.fdtd import cpml_params, updates_with_cpml
from masque import Pattern, shapes
import gridlock
import pcgen


def perturbed_l3(a: float, radius: float, **kwargs) -> Pattern:
    """
    Generate a masque.Pattern object containing a perturbed L3 cavity.

    Args:
        a: Lattice constant.
        radius: Hole radius, in units of a (lattice constant).
        **kwargs: Keyword arguments:
        hole_dose, trench_dose, hole_layer, trench_layer: Shape properties for Pattern.
                Defaults *_dose=1, hole_layer=0, trench_layer=1.
        shifts_a, shifts_r: passed to pcgen.l3_shift; specifies lattice constant (1 -
                multiplicative factor) and radius (multiplicative factor) for shifting
                holes adjacent to the defect (same row). Defaults are 0.15 shift for
                first hole, 0.075 shift for third hole, and no radius change.
        xy_size: [x, y] number of mirror periods in each direction; total size is
                `2 * n + 1` holes in each direction. Default `[10, 10]`.
        perturbed_radius: radius of holes perturbed to form an upwards-driected beam
                (multiplicative factor). Default 1.1.
        trench width: Width of the undercut trenches. Default 1.2e3.

    Return:
        `masque.Pattern` object containing the L3 design
    """

    default_args = {'hole_dose':    1,
                    'trench_dose':  1,
                    'hole_layer':   0,
                    'trench_layer': 1,
                    'shifts_a':     (0.15, 0, 0.075),
                    'shifts_r':     (1.0, 1.0, 1.0),
                    'xy_size':      (10, 10),
                    'perturbed_radius': 1.1,
                    'trench_width': 1.2e3,
                    }
    kwargs = {**default_args, **kwargs}

    xyr = pcgen.l3_shift_perturbed_defect(mirror_dims=kwargs['xy_size'],
                                          perturbed_radius=kwargs['perturbed_radius'],
                                          shifts_a=kwargs['shifts_a'],
                                          shifts_r=kwargs['shifts_r'])
    xyr *= a
    xyr[:, 2] *= radius

    pat = Pattern()
    pat.name = 'L3p-a{:g}r{:g}rp{:g}'.format(a, radius, kwargs['perturbed_radius'])
    pat.shapes += [shapes.Circle(radius=r, offset=(x, y),
                                 dose=kwargs['hole_dose'],
                                 layer=kwargs['hole_layer'])
                   for x, y, r in xyr]

    maxes = numpy.max(numpy.fabs(xyr), axis=0)
    pat.shapes += [shapes.Polygon.rectangle(
        lx=(2 * maxes[0]), ly=kwargs['trench_width'],
        offset=(0, s * (maxes[1] + a + kwargs['trench_width'] / 2)),
        dose=kwargs['trench_dose'], layer=kwargs['trench_layer'])
                   for s in (-1, 1)]
    return pat


def main():
    dtype = numpy.float32
    max_t = 8000            # number of timesteps

    dx = 40                 # discretization (nm/cell)
    pml_thickness = 8       # (number of cells)

    wl = 1550               # Excitation wavelength and fwhm
    dwl = 200

    # Device design parameters
    xy_size = numpy.array([10, 10])
    a = 430
    r = 0.285
    th = 170

    # refractive indices
    n_slab = 3.408  # InGaAsP(80, 50) @ 1550nm
    n_air = 1.0   # air

    # Half-dimensions of the simulation grid
    xy_max = (xy_size + 1) * a * [1, numpy.sqrt(3)/2]
    z_max = 1.6 * a
    xyz_max = numpy.hstack((xy_max, z_max)) + pml_thickness * dx

    # Coordinates of the edges of the cells. The fdtd package can only do square grids at the moment.
    half_edge_coords = [numpy.arange(dx/2, m + dx, step=dx) for m in xyz_max]
    edge_coords = [numpy.hstack((-h[::-1], h)) for h in half_edge_coords]

    # #### Create the grid, mask, and draw the device ####
    grid = gridlock.Grid(edge_coords)
    epsilon = grid.allocate(n_air**2, dtype=dtype)
    grid.draw_slab(epsilon,
                   surface_normal=2,
                   center=[0, 0, 0],
                   thickness=th,
                   eps=n_slab**2)
    mask = perturbed_l3(a, r)

    grid.draw_polygons(epsilon,
                       surface_normal=2,
                       center=[0, 0, 0],
                       thickness=2 * th,
                       eps=n_air**2,
                       polygons=mask.as_polygons())

    print(grid.shape)

    dt = .99/numpy.sqrt(3)
    e = [numpy.zeros_like(epsilon[0], dtype=dtype) for _ in range(3)]
    h = [numpy.zeros_like(epsilon[0], dtype=dtype) for _ in range(3)]

    dxes = [grid.dxyz, grid.autoshifted_dxyz()]

    # PMLs in every direction
    pml_params = [[cpml_params(axis=dd, polarity=pp, dt=dt,
                               thickness=pml_thickness, epsilon_eff=1.0**2)
                   for pp in (-1, +1)]
                  for dd in range(3)]
    update_E, update_H = updates_with_cpml(cpml_params=pml_params, dt=dt,
                                           dxes=dxes, epsilon=epsilon)

    # Source parameters and function
    w = 2 * numpy.pi * dx / wl
    fwhm = dwl * w * w / (2 * numpy.pi * dx)
    alpha = (fwhm ** 2) / 8 * numpy.log(2)
    delay = 7/numpy.sqrt(2 * alpha)

    def field_source(i):
        t0 = i * dt - delay
        return numpy.sin(w * t0) * numpy.exp(-alpha * t0**2)

    # #### Run a bunch of iterations ####
    output_file = h5py.File('simulation_output.h5', 'w')
    start = time.perf_counter()
    for t in range(max_t):
        update_E(e, h, epsilon)

        e[1][tuple(grid.shape//2)] += field_source(t)
        update_H(e, h)

        print('iteration {}: average {} iterations per sec'.format(t, (t+1)/(time.perf_counter()-start)))
        sys.stdout.flush()

        if t % 20 == 0:
            r = sum([(f * f * e).sum() for f, e in zip(e, epsilon)])
            print('E sum', r)

        # Save field slices
        if (t % 20 == 0 and (max_t - t <= 1000 or t <= 2000)) or t == max_t-1:
            print('saving E-field')
            for j, f in enumerate(e):
                output_file['/E{}_t{}'.format('xyz'[j], t)] = f[:, :, round(f.shape[2]/2)]

if __name__ == '__main__':
    main()
