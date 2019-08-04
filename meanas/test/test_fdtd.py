import numpy
import pytest
import dataclasses
from typing import List, Tuple
from numpy.testing import assert_allclose, assert_array_equal

from meanas import fdtd


prng = numpy.random.RandomState(12345)

def assert_fields_close(a, b, *args, **kwargs):
    numpy.testing.assert_allclose(a, b, verbose=False, err_msg='Fields did not match:\n{}\n{}'.format(numpy.rollaxis(a, -1),
                                                                                                      numpy.rollaxis(b, -1)), *args, **kwargs)

def assert_close(a, b, *args, **kwargs):
    numpy.testing.assert_allclose(a, b, *args, **kwargs)


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
        u_hstep = fdtd.energy_hstep(e0=sim.es[ii-1], h1=sim.hs[ii], e2=sim.es[ii],     **args)
        u_estep = fdtd.energy_estep(h0=sim.hs[ii],   e1=sim.es[ii], h2=sim.hs[ii + 1], **args)
        delta_j_A = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii-1], dxes=sim.dxes)
        delta_j_B = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii],   dxes=sim.dxes)

        u += delta_j_A.sum()
        assert_close(u_hstep.sum(), u)
        u += delta_j_B.sum()
        assert_close(u_estep.sum(), u)


def test_poynting_divergence(sim):
    args = {'dxes': sim.dxes,
            'epsilon': sim.epsilon}

    dV = numpy.prod(numpy.meshgrid(*sim.dxes[0], indexing='ij'), axis=0)

    u_eprev = None
    for ii in range(1, 8):
        u_hstep = fdtd.energy_hstep(e0=sim.es[ii-1], h1=sim.hs[ii], e2=sim.es[ii],     **args)
        u_estep = fdtd.energy_estep(h0=sim.hs[ii],   e1=sim.es[ii], h2=sim.hs[ii + 1], **args)
        delta_j_B = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii],   dxes=sim.dxes)

        du_half_h2e = u_estep - u_hstep - delta_j_B
        div_s_h2e = sim.dt * fdtd.poynting_divergence(e=sim.es[ii], h=sim.hs[ii], dxes=sim.dxes) * dV
        assert_fields_close(du_half_h2e, -div_s_h2e)

        if u_eprev is None:
            u_eprev = u_estep
            continue

        # previous half-step
        delta_j_A = fdtd.delta_energy_j(j0=sim.js[ii], e1=sim.es[ii-1], dxes=sim.dxes)

        du_half_e2h = u_hstep - u_eprev - delta_j_A
        div_s_e2h = sim.dt * fdtd.poynting_divergence(e=sim.es[ii-1], h=sim.hs[ii], dxes=sim.dxes) * dV
        assert_fields_close(du_half_e2h, -div_s_e2h)
        u_eprev = u_estep


def test_poynting_planes(sim):
    mask = (sim.js[0] != 0)
    if mask.sum() > 1:
        pytest.skip('test_poynting_planes can only test single point sources')

    args = {'dxes': sim.dxes,
            'epsilon': sim.epsilon}
    dV = numpy.prod(numpy.meshgrid(*sim.dxes[0], indexing='ij'), axis=0)

    mx = numpy.roll(mask, (-1, -1), axis=(0, 1))
    my = numpy.roll(mask, -1, axis=2)
    mz = numpy.roll(mask, (+1, -1), axis=(0, 3))
    px = numpy.roll(mask, -1, axis=0)
    py = mask.copy()
    pz = numpy.roll(mask, +1, axis=0)

    u_eprev = None
    for ii in range(1, 8):
        u_hstep = fdtd.energy_hstep(e0=sim.es[ii-1], h1=sim.hs[ii], e2=sim.es[ii],     **args)
        u_estep = fdtd.energy_estep(h0=sim.hs[ii],   e1=sim.es[ii], h2=sim.hs[ii + 1], **args)

        s_h2e = -fdtd.poynting(e=sim.es[ii], h=sim.hs[ii]) * sim.dt
        s_h2e[0] *= sim.dxes[0][1][None, :, None] * sim.dxes[0][2][None, None, :]
        s_h2e[1] *= sim.dxes[0][0][:, None, None] * sim.dxes[0][2][None, None, :]
        s_h2e[2] *= sim.dxes[0][0][:, None, None] * sim.dxes[0][1][None, :, None]
        planes = [s_h2e[px].sum(), -s_h2e[mx].sum(),
                  s_h2e[py].sum(), -s_h2e[my].sum(),
                  s_h2e[pz].sum(), -s_h2e[mz].sum()]
        assert_close(sum(planes), (u_estep - u_hstep).sum())
        if u_eprev is None:
            u_eprev = u_estep
            continue

        s_e2h = -fdtd.poynting(e=sim.es[ii - 1], h=sim.hs[ii]) * sim.dt
        s_e2h[0] *= sim.dxes[0][1][None, :, None] * sim.dxes[0][2][None, None, :]
        s_e2h[1] *= sim.dxes[0][0][:, None, None] * sim.dxes[0][2][None, None, :]
        s_e2h[2] *= sim.dxes[0][0][:, None, None] * sim.dxes[0][1][None, :, None]
        planes = [s_e2h[px].sum(), -s_e2h[mx].sum(),
                  s_e2h[py].sum(), -s_e2h[my].sum(),
                  s_e2h[pz].sum(), -s_e2h[mz].sum()]
        assert_close(sum(planes), (u_hstep - u_eprev).sum())

        # previous half-step
        u_eprev = u_estep


#####################################
#      Test fixtures
#####################################

@pytest.fixture(scope='module',
                params=[(5, 5, 1),
                        (5, 1, 5),
                        (5, 5, 5),
#                        (7, 7, 7),
                       ])
def shape(request):
    yield (3, *request.param)


@pytest.fixture(scope='module', params=[0.3])
def dt(request):
    yield request.param


@pytest.fixture(scope='module', params=[1.0, 1.5])
def epsilon_bg(request):
    yield request.param


@pytest.fixture(scope='module', params=[1.0, 2.5])
def epsilon_fg(request):
    yield request.param


@pytest.fixture(scope='module', params=['center', '000', 'random'])
def epsilon(request, shape, epsilon_bg, epsilon_fg):
    is3d = (numpy.array(shape) == 1).sum() == 0
    if is3d:
        if request.param == '000':
            pytest.skip('Skipping 000 epsilon because test is 3D (for speed)')
        if epsilon_bg != 1:
            pytest.skip('Skipping epsilon_bg != 1 because test is 3D (for speed)')
        if epsilon_fg not in (1.0, 2.0):
            pytest.skip('Skipping epsilon_fg not in (1, 2) because test is 3D (for speed)')

    epsilon = numpy.full(shape, epsilon_bg, dtype=float)
    if request.param == 'center':
        epsilon[:, shape[1]//2, shape[2]//2, shape[3]//2] = epsilon_fg
    elif request.param == '000':
        epsilon[:, 0, 0, 0] = epsilon_fg
    elif request.param == 'random':
        epsilon[:] = prng.uniform(low=min(epsilon_bg, epsilon_fg),
                                  high=max(epsilon_bg, epsilon_fg),
                                  size=shape)

    yield epsilon


@pytest.fixture(scope='module', params=[1.0])#, 1.5])
def j_mag(request):
    yield request.param


@pytest.fixture(scope='module', params=['center', 'random'])
def j_distribution(request, shape, j_mag):
    j = numpy.zeros(shape)
    if request.param == 'center':
        j[:, shape[1]//2, shape[2]//2, shape[3]//2] = j_mag
    elif request.param == '000':
        j[:, 0, 0, 0] = j_mag
    elif request.param == 'random':
        j[:] = prng.uniform(low=-j_mag, high=j_mag, size=shape)
    yield j


@pytest.fixture(scope='module', params=[1.0, 1.5])
def dx(request):
    yield request.param


@pytest.fixture(scope='module', params=['uniform'])
def dxes(request, shape, dx):
    if request.param == 'uniform':
        dxes = [[numpy.full(s, dx) for s in shape[1:]] for _ in range(2)]
    yield dxes


@pytest.fixture(scope='module',
                params=[(0,),
                        (0, 4, 8),
                       ]
                )
def j_steps(request):
    yield request.param


@dataclasses.dataclass()
class SimResult:
    shape: Tuple[int]
    dt: float
    dxes: List[List[numpy.ndarray]]
    epsilon: numpy.ndarray
    j_distribution: numpy.ndarray
    j_steps: Tuple[int]
    es: List[numpy.ndarray] = dataclasses.field(default_factory=list)
    hs: List[numpy.ndarray] = dataclasses.field(default_factory=list)
    js: List[numpy.ndarray] = dataclasses.field(default_factory=list)


@pytest.fixture(scope='module')
def sim(request, shape, epsilon, dxes, dt, j_distribution, j_steps):
    is3d = (numpy.array(shape) == 1).sum() == 0
    if is3d:
        if dt != 0.3:
            pytest.skip('Skipping dt != 0.3 because test is 3D (for speed)')

    sim = SimResult(
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


