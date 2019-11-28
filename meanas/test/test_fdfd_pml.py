#####################################
# pylint: disable=redefined-outer-name
from typing import List, Tuple
import dataclasses
import pytest
import numpy
from numpy.testing import assert_allclose, assert_array_equal

from .. import fdfd
from ..fdmath import vec, unvec
from .utils import assert_close, assert_fields_close
from .test_fdfd import FDResult


def test_pml(sim, src_polarity):
    dim = numpy.where(numpy.array(sim.shape[1:]) > 1)[0][0]    # Propagation axis

    e_sqr = numpy.squeeze((sim.e.conj() * sim.e).sum(axis=0))

#    from matplotlib import pyplot
#    pyplot.figure()
#    pyplot.plot(numpy.squeeze(e_sqr))
#    pyplot.show(block=True)

    e_sqr_tgt = e_sqr[16:19]
    e_sqr_rev = e_sqr[10:13]
    if src_polarity < 0:
        e_sqr_tgt, e_sqr_rev = e_sqr_rev, e_sqr_tgt

    assert_allclose(e_sqr_rev, 0, atol=1e-12)
    assert_allclose(e_sqr_tgt, 1, rtol=3e-6)


#    pyplot.figure()
#    pyplot.plot(numpy.squeeze(sim.e[0].real), label='Ex_real')
#    pyplot.plot(numpy.squeeze(sim.e[0].imag), label='Ex_imag')
#    pyplot.plot(numpy.squeeze(sim.e[1].real), label='Ey_real')
#    pyplot.plot(numpy.squeeze(sim.e[1].imag), label='Ey_imag')
#    pyplot.plot(numpy.squeeze(sim.e[2].real), label='Ez_real')
#    pyplot.plot(numpy.squeeze(sim.e[2].imag), label='Ez_imag')
#    pyplot.legend()
#    pyplot.show(block=True)


#      Test fixtures
#####################################
# Also see conftest.py

@pytest.fixture(params=[1/1500])
def omega(request):
    yield request.param


@pytest.fixture(params=[None])
def pec(request):
    yield request.param


@pytest.fixture(params=[None])
def pmc(request):
    yield request.param



@pytest.fixture(params=[(30, 1, 1),
                        (1, 30, 1),
                        (1, 1, 30)])
def shape(request):
    yield (3, *request.param)


@pytest.fixture(params=[+1, -1])
def src_polarity(request):
    yield request.param


@pytest.fixture()
def j_distribution(request, shape, epsilon, dxes, omega, src_polarity):
    j = numpy.zeros(shape, dtype=complex)

    dim = numpy.where(numpy.array(shape[1:]) > 1)[0][0]    # Propagation axis
    other_dims = [0, 1, 2]
    other_dims.remove(dim)

    dx_prop = (dxes[0][dim][shape[dim + 1] // 2] +
               dxes[1][dim][shape[dim + 1] // 2]) / 2       #TODO is this right for nonuniform dxes?

    # Mask only contains components orthogonal to propagation direction
    center_mask = numpy.zeros(shape, dtype=bool)
    center_mask[other_dims, shape[1]//2, shape[2]//2, shape[3]//2] = True
    if (epsilon[center_mask] != epsilon[center_mask][0]).any():
        center_mask[other_dims[1]] = False          # If epsilon is not isotropic, pick only one dimension


    wavenumber = omega * numpy.sqrt(epsilon[center_mask].mean())
    wavenumber_corrected = 2 / dx_prop * numpy.arcsin(wavenumber * dx_prop / 2)

    e = numpy.zeros_like(epsilon, dtype=complex)
    e[center_mask] = 1 / numpy.linalg.norm(center_mask[:])

    slices = [slice(None), slice(None), slice(None)]
    slices[dim] = slice(shape[dim + 1] // 2,
                        shape[dim + 1] // 2 + 1)

    j = fdfd.waveguide_3d.compute_source(E=e, wavenumber=wavenumber_corrected, omega=omega, dxes=dxes,
                                         axis=dim, polarity=src_polarity, slices=slices, epsilon=epsilon)
    yield j


@pytest.fixture()
def epsilon(request, shape, epsilon_bg, epsilon_fg):
    epsilon = numpy.full(shape, epsilon_fg, dtype=float)
    yield epsilon


@pytest.fixture(params=['uniform'])
def dxes(request, shape, dx, omega, epsilon_fg):
    if request.param == 'uniform':
        dxes = [[numpy.full(s, dx) for s in shape[1:]] for _ in range(2)]
    dim = numpy.where(numpy.array(shape[1:]) > 1)[0][0]    # Propagation axis
    for axis in (dim,):
        for polarity in (-1, 1):
            dxes = fdfd.scpml.stretch_with_scpml(dxes, axis=axis, polarity=polarity,
                                                 omega=omega, epsilon_effective=epsilon_fg,
                                                 thickness=10)
    yield dxes


@pytest.fixture()
def sim(request, shape, epsilon, dxes, j_distribution, omega, pec, pmc):
    j_vec = vec(j_distribution)
    eps_vec = vec(epsilon)
    e_vec = fdfd.solvers.generic(J=j_vec, omega=omega, dxes=dxes, epsilon=eps_vec,
                                 matrix_solver_opts={'atol': 1e-15, 'tol': 1e-11})
    e = unvec(e_vec, shape[1:])

    sim = FDResult(
        shape=shape,
        dxes=dxes,
        epsilon=epsilon,
        j=j_distribution,
        e=e,
        pec=pec,
        pmc=pmc,
        omega=omega,
        )

    return sim
