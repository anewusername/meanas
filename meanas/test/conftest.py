from typing import List, Tuple
import numpy
import pytest

from .utils import PRNG

#####################################
#      Test fixtures
#####################################

@pytest.fixture(scope='module',
                params=[(5, 5, 1),
                        (5, 1, 5),
                        (5, 5, 5),
                        #(7, 7, 7),
                       ])
def shape(request):
    yield (3, *request.param)


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
        epsilon[:] = PRNG.uniform(low=min(epsilon_bg, epsilon_fg),
                                  high=max(epsilon_bg, epsilon_fg),
                                  size=shape)

    yield epsilon


@pytest.fixture(scope='module', params=[1.0])#, 1.5])
def j_mag(request):
    yield request.param


@pytest.fixture(scope='module', params=[1.0, 1.5])
def dx(request):
    yield request.param


@pytest.fixture(scope='module', params=['uniform'])
def dxes(request, shape, dx):
    if request.param == 'uniform':
        dxes = [[numpy.full(s, dx) for s in shape[1:]] for _ in range(2)]
    yield dxes
