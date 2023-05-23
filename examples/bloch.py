import numpy, scipy, gridlock, meanas
from meanas.fdfd import bloch
from numpy.linalg import norm
import logging
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

WISDOM_FILEPATH = Path.home() / '.local/share/pyfftw/wisdom.pickle'


def pyfftw_save_wisdom(path):
    path = Path(path)
    try:
        import pyfftw
        import pickle
    except ImportError as e:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    wisdom = pyfftw.export_wisdom()
    with open(path, 'wb') as f:
        pickle.dump(wisdom, f)


def pyfftw_load_wisdom(path):
    path = Path(path)
    try:
        import pyfftw
        import pickle
    except ImportError as e:
        pass

    if path.exists():
        with open(path, 'rb') as f:
            wisdom = pickle.load(f)
        pyfftw.import_wisdom(wisdom)


logger.info('Drawing grid...')
dx = 40
x_period = 400
y_period = z_period = 2000
g = gridlock.Grid([
    numpy.arange(-x_period/2, x_period/2, dx),
    numpy.arange(-1000, 1000, dx),
    numpy.arange(-1000, 1000, dx)],
    shifts=numpy.array([[0,0,0]]),
    periodic=True,
    )
gdata = g.allocate(1.445**2)

g.draw_cuboid(gdata, [0,0,0], [200e8, 220, 220], foreground=3.47**2)

#x_period = y_period = z_period = 13000
#g = gridlock.Grid([numpy.arange(3), ]*3,
#                  shifts=numpy.array([[0, 0, 0]]),
#                  initial=2.0**2,
#                  periodic=True)

g2 = g.copy()
g2.shifts = numpy.zeros((6,3))
g2data = g2.allocate(0)

epsilon = [gdata[0],] * 3
reciprocal_lattice = numpy.diag(1000/numpy.array([x_period, y_period, z_period])) #cols are vectors

pyfftw_load_wisdom(WISDOM_FILEPATH)

#print('Finding k at 1550nm')
#k, f = bloch.find_k(frequency=1000/1550,
#                    tolerance=(1000 * (1/1550 - 1/1551)),
#                    direction=[1, 0, 0],
#                    G_matrix=reciprocal_lattice,
#                    epsilon=epsilon,
#                    band=0)
#
#kf = norm(reciprocal_lattice @ k) / f)
#print(f'{k=}, {f=}, 1/f={1/f}, k/f={kf}')

logger.info('Finding f at [0.25, 0, 0]')
for k0x in [.25]:
    k0 = numpy.array([k0x, 0, 0])

    kmag = norm(reciprocal_lattice @ k0)
    tolerance = (1000/1550) * 1e-4/1.5  # df = f * dn_eff / n
    logger.info(f'tolerance {tolerance}')

    n, v = bloch.eigsolve(4, k0, G_matrix=reciprocal_lattice, epsilon=epsilon, tolerance=tolerance**2)
    v2e = bloch.hmn_2_exyz(k0, G_matrix=reciprocal_lattice, epsilon=epsilon)
    v2h = bloch.hmn_2_hxyz(k0, G_matrix=reciprocal_lattice, epsilon=epsilon)
    ki = bloch.generate_kmn(k0, reciprocal_lattice, g.shape)

    z = 0
    e = v2e(v[0])
    for i in range(3):
        g2data[i] += numpy.real(e[i])
        g2data[i+3] += numpy.imag(e[i])

    f = numpy.sqrt(numpy.real(numpy.abs(n))) # TODO
    print(f'{k0x=:3g}')
    print(f'eigval={n}')
    print(f'{f=}')
    n_eff = norm(reciprocal_lattice @ k0) / f
    print(f'kmag/f = n_eff = {n_eff}')
    print(f'wl={1/f}\n')

pyfftw_save_wisdom(WISDOM_FILEPATH)
