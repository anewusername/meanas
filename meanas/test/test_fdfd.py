from typing import List, Tuple
import dataclasses
import pytest
import numpy
from numpy.testing import assert_allclose, assert_array_equal

from .. import fdfd, vec, unvec
from .utils import assert_close, assert_fields_close


def test_poynting_planes(sim):
    mask = (sim.j != 0).any(axis=0)
    if mask.sum() > 1:
        pytest.skip(f'test_poynting_planes can only test single point sources, got {mask.sum()}')

    mx = numpy.roll(mask, -1, axis=0)
    my = numpy.roll(mask, -1, axis=1)
    mz = numpy.roll(mask, -1, axis=2)

    e2h = fdfd.operators.e2h(omega=sim.omega, dxes=sim.dxes, pmc=sim.pmc)
    ev = vec(sim.e)
    hv = e2h @ ev

    exh = fdfd.operators.poynting_e_cross(e=ev, dxes=sim.dxes) @ hv.conj()
    s = unvec(exh.real / 2, sim.shape[1:])
    planes = [s[0, mask].sum(), -s[0, mx].sum(),
              s[0, mask].sum(), -s[1, my].sum(),
              s[0, mask].sum(), -s[2, mz].sum()]

    e_dot_j = sim.e * sim.j
    src_energy = -e_dot_j.real / 2

    assert_close(sum(planes), (src_energy).sum())


#####################################
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


@dataclasses.dataclass()
class SimResult:
    shape: Tuple[int]
    dxes: List[List[numpy.ndarray]]
    epsilon: numpy.ndarray
    omega: complex
    j: numpy.ndarray
    e: numpy.ndarray
    pmc: numpy.ndarray
    pec: numpy.ndarray

@pytest.fixture()
def sim(request, shape, epsilon, dxes, j_distribution, omega, pec, pmc):
#    is3d = (numpy.array(shape) == 1).sum() == 0
#    if is3d:
#        pytest.skip('Skipping dt != 0.3 because test is 3D (for speed)')

    j_vec = vec(j_distribution)
    eps_vec = vec(epsilon)
    e_vec = fdfd.solvers.generic(J=j_vec, omega=omega, dxes=dxes, epsilon=eps_vec)
    e = unvec(e_vec, shape[1:])

    sim = SimResult(
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

