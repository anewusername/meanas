# ruff: noqa: ARG001
import dataclasses
import pytest       # type: ignore
import numpy
from numpy.typing import NDArray
#from numpy.testing import assert_allclose, assert_array_equal

from .. import fdfd
from ..fdmath import vec, unvec
from .utils import assert_close  # , assert_fields_close
from .conftest import FixtureRequest


def test_residual(sim: 'FDResult') -> None:
    A = fdfd.operators.e_full(sim.omega, sim.dxes, vec(sim.epsilon)).tocsr()
    b = -1j * sim.omega * vec(sim.j)
    residual = A @ vec(sim.e) - b
    assert numpy.linalg.norm(residual) < 1e-10


def test_poynting_planes(sim: 'FDResult') -> None:
    mask = (sim.j != 0).any(axis=0)
    if mask.sum() != 2:
        pytest.skip(f'test_poynting_planes will only test 2-point sources, got {mask.sum()}')
#    for dxg in sim.dxes:
#        for dxa in dxg:
#            if not (dxa == sim.dxes[0][0][0]).all():
#                pytest.skip('test_poynting_planes skips nonuniform dxes')

    # pick only the second point
    points = numpy.where(mask)
    mask[points[0][0], points[1][0], points[2][0]] = 0

    mx = numpy.roll(mask, -1, axis=0)
    my = numpy.roll(mask, -1, axis=1)
    mz = numpy.roll(mask, -1, axis=2)

    e2h = fdfd.operators.e2h(omega=sim.omega, dxes=sim.dxes, pmc=sim.pmc)
    ev = vec(sim.e)
    hv = e2h @ ev

    exh = fdfd.operators.poynting_e_cross(e=ev, dxes=sim.dxes) @ hv.conj()
    s = unvec(exh.real / 2, sim.shape[1:])
    planes = [s[0, mask].sum(), -s[0, mx].sum(),
              s[1, mask].sum(), -s[1, my].sum(),
              s[2, mask].sum(), -s[2, mz].sum()]

    e_dot_j = sim.e * sim.j.conj()
    dv = (sim.dxes[0][0][:, None, None]
        * sim.dxes[0][1][None, :, None]
        * sim.dxes[0][2][None, None, :]
        )
    src_energy = -(e_dot_j.real * dv)[:, mask] / 2

    assert_close(sum(planes), src_energy.sum(), rtol=1e-6)      # TODO improve energy calculation accuracy?


#####################################
#      Test fixtures
#####################################
# Also see conftest.py

@pytest.fixture(params=[1 / 1500])
def omega(request: FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[None])
def pec(request: FixtureRequest) -> NDArray[numpy.float64] | None:
    return request.param


@pytest.fixture(params=[None])
def pmc(request: FixtureRequest) -> NDArray[numpy.float64] | None:
    return request.param


#@pytest.fixture(scope='module',
#                params=[(25, 5, 5)])
#def shape(request: FixtureRequest):
#    return (3, *request.param)


@pytest.fixture(params=['diag'])        # 'center'
def j_distribution(
        request: FixtureRequest,
        shape: tuple[int, ...],
        j_mag: float,
        ) -> NDArray[numpy.float64]:
    j = numpy.zeros(shape, dtype=complex)
    center_mask = numpy.zeros(shape, dtype=bool)
    center_mask[:, shape[1] // 2, shape[2] // 2, shape[3] // 2] = True

    if request.param == 'center':
        j[center_mask] = j_mag
    elif request.param == 'diag':
        j[numpy.roll(center_mask, [1, 1, 1], axis=(1, 2, 3))] = (1 + 1j) * j_mag
        j[numpy.roll(center_mask, [-1, -1, -1], axis=(1, 2, 3))] = (1 - 1j) * j_mag
    return j


@dataclasses.dataclass()
class FDResult:
    shape: tuple[int, ...]
    dxes: list[list[NDArray[numpy.float64]]]
    epsilon: NDArray[numpy.float64]
    omega: complex
    j: NDArray[numpy.complex128]
    e: NDArray[numpy.complex128]
    pmc: NDArray[numpy.float64] | None
    pec: NDArray[numpy.float64] | None


@pytest.fixture()
def sim(
        request: FixtureRequest,
        shape: tuple[int, ...],
        epsilon: NDArray[numpy.float64],
        dxes: list[list[NDArray[numpy.float64]]],
        j_distribution: NDArray[numpy.complex128],
        omega: float,
        pec: NDArray[numpy.float64] | None,
        pmc: NDArray[numpy.float64] | None,
        ) -> FDResult:
    """
    Build simulation from parts
    """
#    is3d = (numpy.array(shape) == 1).sum() == 0
#    if is3d:
#        pytest.skip('Skipping dt != 0.3 because test is 3D (for speed)')

#    # If no edge currents, add pmls
#    src_mask = j_distribution.any(axis=0)
#    th = 10
#    #if src_mask.sum() - src_mask[th:-th, th:-th, th:-th].sum() == 0:
#    if src_mask.sum() - src_mask[th:-th, :, :].sum() == 0:
#        for axis in (0,):
#            for polarity in (-1, 1):
#                dxes = fdfd.scpml.stretch_with_scpml(dxes, axis=axis, polarity=polarity,

    j_vec = vec(j_distribution)
    eps_vec = vec(epsilon)
    e_vec = fdfd.solvers.generic(
        J=j_vec,
        omega=omega,
        dxes=dxes,
        epsilon=eps_vec,
        matrix_solver_opts={'atol': 1e-15, 'rtol': 1e-11},
        )
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
