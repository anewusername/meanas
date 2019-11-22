# pylint: disable=redefined-outer-name, no-member
from typing import List, Tuple
import dataclasses
import pytest
import numpy
from numpy.testing import assert_allclose, assert_array_equal

from .. import fdtd
from .utils import assert_close, assert_fields_close, PRNG


def test_initial_fields(sim):
    # Make sure initial fields didn't change
    e0 = sim.es[0]
    h0 = sim.hs[0]
    j0 = sim.js[0]
    mask = (j0 != 0)

    assert_fields_close(e0[mask], j0[mask] / sim.epsilon[mask])
    assert not e0[~mask].any()
    assert not h0.any()


def test_initial_energy(sim):
    """
    Assumes fields start at 0 before J0 is added
    """
    j0 = sim.js[0]
    e0 = sim.es[0]
    h0 = sim.hs[0]
    h1 = sim.hs[1]
    mask = (j0 != 0)
    dV = numpy.prod(numpy.meshgrid(*sim.dxes[0], indexing='ij'), axis=0)
    u0 = (j0 * j0.conj() / sim.epsilon * dV).sum(axis=0)
    args = {'dxes': sim.dxes,
            'epsilon': sim.epsilon}

    # Make sure initial energy and E dot J are correct
    energy0 = fdtd.energy_estep(h0=h0, e1=e0, h2=h1, **args)
    e0_dot_j0 = fdtd.delta_energy_j(j0=j0, e1=e0, dxes=sim.dxes)
    assert_fields_close(energy0, u0)
    assert_fields_close(e0_dot_j0, u0)


def test_energy_conservation(sim):
    """
    Assumes fields start at 0 before J0 is added
    """
    e0 = sim.es[0]
    j0 = sim.js[0]
    u = fdtd.delta_energy_j(j0=j0, e1=e0, dxes=sim.dxes).sum()
    args = {'dxes': sim.dxes,
            'epsilon': sim.epsilon}

    for ii in range(1, 8):
        u_hstep = fdtd.energy_hstep(e0=sim.es[ii-1], h1=sim.hs[ii], e2=sim.es[ii],     **args)  # pylint: disable=bad-whitespace
        u_estep = fdtd.energy_estep(h0=sim.hs[ii],   e1=sim.es[ii], h2=sim.hs[ii + 1], **args)  # pylint: disable=bad-whitespace
        delta_j_A = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii-1], dxes=sim.dxes)
        delta_j_B = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii],   dxes=sim.dxes)  # pylint: disable=bad-whitespace

        u += delta_j_A.sum()
        assert_close(u_hstep.sum(), u)
        u += delta_j_B.sum()
        assert_close(u_estep.sum(), u)


def test_poynting_divergence(sim):
    args = {'dxes': sim.dxes,
            'epsilon': sim.epsilon}

    u_eprev = None
    for ii in range(1, 8):
        u_hstep = fdtd.energy_hstep(e0=sim.es[ii-1], h1=sim.hs[ii], e2=sim.es[ii],     **args)  # pylint: disable=bad-whitespace
        u_estep = fdtd.energy_estep(h0=sim.hs[ii],   e1=sim.es[ii], h2=sim.hs[ii + 1], **args)  # pylint: disable=bad-whitespace
        delta_j_B = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii], dxes=sim.dxes)

        du_half_h2e = u_estep - u_hstep - delta_j_B
        div_s_h2e = sim.dt * fdtd.poynting_divergence(e=sim.es[ii], h=sim.hs[ii], dxes=sim.dxes)
        assert_fields_close(du_half_h2e, -div_s_h2e)

        if u_eprev is None:
            u_eprev = u_estep
            continue

        # previous half-step
        delta_j_A = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii-1], dxes=sim.dxes)

        du_half_e2h = u_hstep - u_eprev - delta_j_A
        div_s_e2h = sim.dt * fdtd.poynting_divergence(e=sim.es[ii-1], h=sim.hs[ii], dxes=sim.dxes)
        assert_fields_close(du_half_e2h, -div_s_e2h)
        u_eprev = u_estep


def test_poynting_planes(sim):
    mask = (sim.js[0] != 0).any(axis=0)
    if mask.sum() > 1:
        pytest.skip('test_poynting_planes can only test single point sources, got {}'.format(mask.sum()))

    args = {'dxes': sim.dxes,
            'epsilon': sim.epsilon}

    mx = numpy.roll(mask, -1, axis=0)
    my = numpy.roll(mask, -1, axis=1)
    mz = numpy.roll(mask, -1, axis=2)

    u_eprev = None
    for ii in range(1, 8):
        u_hstep = fdtd.energy_hstep(e0=sim.es[ii-1], h1=sim.hs[ii], e2=sim.es[ii],     **args)  # pylint: disable=bad-whitespace
        u_estep = fdtd.energy_estep(h0=sim.hs[ii],   e1=sim.es[ii], h2=sim.hs[ii + 1], **args)  # pylint: disable=bad-whitespace
        delta_j_B = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii], dxes=sim.dxes)
        du_half_h2e = u_estep - u_hstep - delta_j_B

        s_h2e = -fdtd.poynting(e=sim.es[ii], h=sim.hs[ii], dxes=sim.dxes) * sim.dt
        planes = [s_h2e[0, mask].sum(), -s_h2e[0, mx].sum(),
                  s_h2e[1, mask].sum(), -s_h2e[1, my].sum(),
                  s_h2e[2, mask].sum(), -s_h2e[2, mz].sum()]

        assert_close(sum(planes), du_half_h2e[mask])

        if u_eprev is None:
            u_eprev = u_estep
            continue

        delta_j_A = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii-1], dxes=sim.dxes)
        du_half_e2h = u_hstep - u_eprev - delta_j_A

        s_e2h = -fdtd.poynting(e=sim.es[ii - 1], h=sim.hs[ii], dxes=sim.dxes) * sim.dt
        planes = [s_e2h[0, mask].sum(), -s_e2h[0, mx].sum(),
                  s_e2h[1, mask].sum(), -s_e2h[1, my].sum(),
                  s_e2h[2, mask].sum(), -s_e2h[2, mz].sum()]
        assert_close(sum(planes), du_half_e2h[mask])

        # previous half-step
        u_eprev = u_estep


#####################################
#      Test fixtures
#####################################
# Also see conftest.py


@pytest.fixture(params=[0.3])
def dt(request):
    yield request.param


@dataclasses.dataclass()
class TDResult:
    shape: Tuple[int]
    dt: float
    dxes: List[List[numpy.ndarray]]
    epsilon: numpy.ndarray
    j_distribution: numpy.ndarray
    j_steps: Tuple[int]
    es: List[numpy.ndarray] = dataclasses.field(default_factory=list)
    hs: List[numpy.ndarray] = dataclasses.field(default_factory=list)
    js: List[numpy.ndarray] = dataclasses.field(default_factory=list)


@pytest.fixture(params=[(0, 4, 8),]) #(0,)])
def j_steps(request):
    yield request.param


@pytest.fixture(params=['center', 'random'])
def j_distribution(request, shape, j_mag):
    j = numpy.zeros(shape)
    if request.param == 'center':
        j[:, shape[1]//2, shape[2]//2, shape[3]//2] = j_mag
    elif request.param == '000':
        j[:, 0, 0, 0] = j_mag
    elif request.param == 'random':
        j[:] = PRNG.uniform(low=-j_mag, high=j_mag, size=shape)
    yield j


@pytest.fixture()
def sim(request, shape, epsilon, dxes, dt, j_distribution, j_steps):
    is3d = (numpy.array(shape) == 1).sum() == 0
    if is3d:
        if dt != 0.3:
            pytest.skip('Skipping dt != 0.3 because test is 3D (for speed)')

    sim = TDResult(
        shape=shape,
        dt=dt,
        dxes=dxes,
        epsilon=epsilon,
        j_distribution=j_distribution,
        j_steps=j_steps,
        )

    e = numpy.zeros_like(epsilon)
    h = numpy.zeros_like(epsilon)

    assert 0 in j_steps
    j_zeros = numpy.zeros_like(j_distribution)

    eh2h = fdtd.maxwell_h(dt=dt, dxes=dxes)
    eh2e = fdtd.maxwell_e(dt=dt, dxes=dxes)
    for tt in range(10):
        e = e.copy()
        h = h.copy()
        eh2h(e, h)
        eh2e(e, h, epsilon)
        if tt in j_steps:
            e += j_distribution / epsilon
            sim.js.append(j_distribution)
        else:
            sim.js.append(j_zeros)
        sim.es.append(e)
        sim.hs.append(h)
    return sim
