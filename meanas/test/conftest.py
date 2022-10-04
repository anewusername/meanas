"""

Test fixtures

"""
from typing import Tuple, Iterable, List, Any
import numpy
from numpy.typing import NDArray, ArrayLike
import pytest       # type: ignore

from .utils import PRNG


FixtureRequest = Any


@pytest.fixture(scope='module',
                params=[(5, 5, 1),
                        (5, 1, 5),
                        (5, 5, 5),
                        # (7, 7, 7),
                       ])
def shape(request: FixtureRequest) -> Iterable[Tuple[int, ...]]:
    yield (3, *request.param)


@pytest.fixture(scope='module', params=[1.0, 1.5])
def epsilon_bg(request: FixtureRequest) -> Iterable[float]:
    yield request.param


@pytest.fixture(scope='module', params=[1.0, 2.5])
def epsilon_fg(request: FixtureRequest) -> Iterable[float]:
    yield request.param


@pytest.fixture(scope='module', params=['center', '000', 'random'])
def epsilon(
        request: FixtureRequest,
        shape: Tuple[int, ...],
        epsilon_bg: float,
        epsilon_fg: float,
        ) -> Iterable[NDArray[numpy.float64]]:
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
        epsilon[:, shape[1] // 2, shape[2] // 2, shape[3] // 2] = epsilon_fg
    elif request.param == '000':
        epsilon[:, 0, 0, 0] = epsilon_fg
    elif request.param == 'random':
        epsilon[:] = PRNG.uniform(low=min(epsilon_bg, epsilon_fg),
                                  high=max(epsilon_bg, epsilon_fg),
                                  size=shape)

    yield epsilon


@pytest.fixture(scope='module', params=[1.0])  # 1.5
def j_mag(request: FixtureRequest) -> Iterable[float]:
    yield request.param


@pytest.fixture(scope='module', params=[1.0, 1.5])
def dx(request: FixtureRequest) -> Iterable[float]:
    yield request.param


@pytest.fixture(scope='module', params=['uniform', 'centerbig'])
def dxes(
        request: FixtureRequest,
        shape: Tuple[int, ...],
        dx: float,
        ) -> Iterable[List[List[NDArray[numpy.float64]]]]:
    if request.param == 'uniform':
        dxes = [[numpy.full(s, dx) for s in shape[1:]] for _ in range(2)]
    elif request.param == 'centerbig':
        dxes = [[numpy.full(s, dx) for s in shape[1:]] for _ in range(2)]
        for eh in (0, 1):
            for ax in (0, 1, 2):
                dxes[eh][ax][dxes[eh][ax].size // 2] *= 1.1
    elif request.param == 'random':
        dxe = [PRNG.uniform(low=1.0 * dx, high=1.1 * dx, size=s) for s in shape[1:]]
        dxh = [(d + numpy.roll(d, -1)) / 2 for d in dxe]
        dxes = [dxe, dxh]
    yield dxes

