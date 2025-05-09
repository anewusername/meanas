"""
Bloch eigenmode solver/operators

This module contains functions for generating and solving the
 3D Bloch eigenproblem. The approach is to transform the problem
 into the (spatial) fourier domain, transforming the equation

    1/mu * curl(1/eps * curl(H_eigenmode)) = (w/c)^2 H_eigenmode

 into

    conv(1/mu_k, ik x conv(1/eps_k, ik x H_k)) = (w/c)^2 H_k

 where:

  - the `_k` subscript denotes a 3D fourier transformed field
  - each component of `H_k` corresponds to a plane wave with wavevector `k`
  - `x` is the cross product
  - `conv()` denotes convolution

 Since `k` and `H` are orthogonal for each plane wave, we can use each
 `k` to create an orthogonal basis (k, m, n), with `k x m = n`, and
 `|m| = |n| = 1`. The cross products are then simplified as follows:

  - `h` is shorthand for `H_k`
  - `(...)_xyz` denotes the `(x, y, z)` basis
  - `(...)_kmn` denotes the `(k, m, n)` basis
  - `hm` is the component of `h` in the `m` direction, etc.

  We know

    k @ h = kx hx + ky hy + kz hz = 0 = hk
    h = hk + hm + hn = hm + hn
    k = kk + km + kn = kk = |k|

  We can write

    k x h = (ky hz - kz hy,
             kz hx - kx hz,
             kx hy - ky hx)_xyz
          = ((k x h) @ k, (k x h) @ m, (k x h) @ n)_kmn
          = (0, (m x k) @ h, (n x k) @ h)_kmn         # triple product ordering
          = (0, kk (-n @ h), kk (m @ h))_kmn          # (m x k) = -|k| n, etc.
          = |k| (0, -h @ n, h @ m)_kmn

  which gives us a straightforward way to perform the cross product
  while simultaneously transforming into the `_kmn` basis.
  We can also write

    k x h = (km hn - kn hm,
             kn hk - kk hn,
             kk hm - km hk)_kmn
          = (0, -kk hn, kk hm)_kmn
          = (-kk hn)(mx, my, mz)_xyz + (kk hm)(nx, ny, nz)_xyz
          = |k| (hm * (nx, ny, nz)_xyz
               - hn * (mx, my, mz)_xyz)

  which gives us a way to perform the cross product while simultaneously
  trasnforming back into the `_xyz` basis.

 We can also simplify `conv(X_k, Y_k)` as `fftn(X * ifftn(Y_k))`.

 Using these results and storing `H_k` as `h = (hm, hn)`, we have

    e_xyz = fftn(1/eps * ifftn(|k| (hm * n - hn * m)))
    b_mn = |k| (-e_xyz @ n, e_xyz @ m)
    h_mn = fftn(1/mu * ifftn(b_m * m + b_n * n))

 which forms the operator from the left side of the equation.

 We can then use a preconditioned block Rayleigh iteration algorithm, as in
  SG Johnson and JD Joannopoulos, Block-iterative frequency-domain methods
  for Maxwell's equations in a planewave basis, Optics Express 8, 3, 173-190 (2001)
 (similar to that used in MPB) to find the eigenvectors for this operator.

 ===

 Typically you will want to do something like

    recip_lattice = numpy.diag(1/numpy.array(epsilon[0].shape * dx))
    n, v = bloch.eigsolve(5, k0, recip_lattice, epsilon)
    f = numpy.sqrt(-numpy.real(n[0]))
    n_eff = norm(recip_lattice @ k0) / f

    v2e = bloch.hmn_2_exyz(k0, recip_lattice, epsilon)
    e_field = v2e(v[0])

    k, f = find_k(frequency=1/1550,
                  tolerance=(1/1550 - 1/1551),
                  direction=[1, 0, 0],
                  G_matrix=recip_lattice,
                  epsilon=epsilon,
                  band=0)

"""

from typing import Any, cast
from collections.abc import Callable, Sequence
import logging
import numpy
from numpy import pi, real, trace
from numpy.fft import fftfreq
from numpy.typing import NDArray, ArrayLike
import scipy
import scipy.optimize
from scipy.linalg import norm
import scipy.sparse.linalg as spalg

from ..fdmath import fdfield_t, cfdfield_t


logger = logging.getLogger(__name__)


try:
    import pyfftw.interfaces.numpy_fft  # type: ignore
    import pyfftw.interfaces            # type: ignore
    logger.info('Using pyfftw')

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(3600)
    fftw_args = {
        #'threads': multiprocessing.cpu_count(),
        'overwrite_input': True,
        #'planner_effort': 'FFTW_PATIENT',
        }

    def fftn(*args: Any, **kwargs: Any) -> NDArray[numpy.complex128]:
        return pyfftw.interfaces.numpy_fft.fftn(*args, **kwargs, **fftw_args)

    def ifftn(*args: Any, **kwargs: Any) -> NDArray[numpy.complex128]:
        return pyfftw.interfaces.numpy_fft.ifftn(*args, **kwargs, **fftw_args)

except ImportError:
    from numpy.fft import fftn, ifftn
    logger.info('Using numpy fft')


def generate_kmn(
        k0: ArrayLike,
        G_matrix: ArrayLike,
        shape: Sequence[int],
        ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
    """
    Generate a (k, m, n) orthogonal basis for each k-vector in the simulation grid.

    Args:
        k0: [k0x, k0y, k0z], Bloch wavevector, in G basis.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        shape: [nx, ny, nz] shape of the simulation grid.

    Returns:
        `(|k|, m, n)` where `|k|` has shape `tuple(shape) + (1,)`
            and `m`, `n` have shape `tuple(shape) + (3,)`.
            All are given in the xyz basis (e.g. `|k|[0,0,0] = norm(G_matrix @ k0)`).
    """
    k0 = numpy.array(k0)
    G_matrix = numpy.asarray(G_matrix)

    Gi_grids = numpy.array(numpy.meshgrid(*(fftfreq(n, 1 / n) for n in shape[:3]), indexing='ij'))
    Gi = numpy.moveaxis(Gi_grids, 0, -1)

    k_G = k0[None, None, None, :] - Gi
    k_xyz = numpy.moveaxis(G_matrix @ numpy.moveaxis(k_G, 3, 2), 3, 2)

    m = numpy.broadcast_to([0, 1, 0], tuple(shape[:3]) + (3,)).astype(float)
    n = numpy.broadcast_to([0, 0, 1], tuple(shape[:3]) + (3,)).astype(float)

    xy_non0 = numpy.any(k_xyz[:, :, :, 0:1] != 0, axis=3)
    if numpy.any(xy_non0):
        u = numpy.cross(k_xyz[xy_non0], [0, 0, 1])
        m[xy_non0, :] = u / norm(u, axis=1)[:, None]

    z_non0 = numpy.any(k_xyz != 0, axis=3)
    if numpy.any(z_non0):
        v = numpy.cross(k_xyz[z_non0], m[z_non0])
        n[z_non0, :] = v / norm(v, axis=1)[:, None]

    k_mag = norm(k_xyz, axis=3)[:, :, :, None]
    return k_mag, m, n


def maxwell_operator(
        k0: ArrayLike,
        G_matrix: ArrayLike,
        epsilon: fdfield_t,
        mu: fdfield_t | None = None
        ) -> Callable[[NDArray[numpy.complex128]], NDArray[numpy.complex128]]:
    """
    Generate the Maxwell operator

        conv(1/mu_k, ik x conv(1/eps_k, ik x ___))

    which is the spatial-frequency-space representation of

        1/mu * curl(1/eps * curl(___))

    The operator is a function that acts on a vector h_mn of size `2 * epsilon[0].size`

    See the `meanas.fdfd.bloch` docstring for more information.

    Args:
        k0: Bloch wavevector, `[k0x, k0y, k0z]`.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        epsilon: Dielectric constant distribution for the simulation.
                 All fields are sampled at cell centers (i.e., NOT Yee-gridded)
        mu: Magnetic permability distribution for the simulation.
            Default None (1 everywhere).

    Returns:
        Function which applies the maxwell operator to h_mn.
    """

    shape = epsilon[0].shape + (1,)
    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    epsilon = numpy.moveaxis(epsilon, 0, -1)
    if mu is not None:
        mu = numpy.moveaxis(mu, 0, -1)

    def operator(h: NDArray[numpy.complex128]) -> NDArray[numpy.complex128]:
        """
        Maxwell operator for Bloch eigenmode simulation.

        h is complex 2-field in (m, n) basis, vectorized

        Args:
            h: Raveled h_mn; size `2 * epsilon[0].size`.
                Altered in-place.

        Returns:
            Raveled conv(1/mu_k, ik x conv(1/eps_k, ik x h_mn)), returned
            and overwritten in-place of `h`.
        """
        hin_m, hin_n = (hi.reshape(shape) for hi in numpy.split(h, 2))

        #{d,e,h}_xyz fields are complex 3-fields in (1/x, 1/y, 1/z) basis

        # cross product and transform into xyz basis
        d_xyz = (n * hin_m
               - m * hin_n) * k_mag         # noqa: E128

        # divide by epsilon
        temp = ifftn(d_xyz, axes=range(3))   # reuses d_xyz if using pyfftw
        temp /= epsilon
        e_xyz = fftn(temp, axes=range(3))

        # cross product and transform into mn basis
        b_m = numpy.sum(e_xyz * n, axis=3, keepdims=True) * -k_mag
        b_n = numpy.sum(e_xyz * m, axis=3, keepdims=True) * +k_mag

        if mu is None:
            h_m, h_n = b_m, b_n
        else:
            # transform from mn to xyz
            b_xyz = (m * b_m[:, :, :, None]
                   + n * b_n[:, :, :, None])    # noqa: E128

            # divide by mu
            temp = ifftn(b_xyz, axes=range(3))
            temp /= mu
            h_xyz = fftn(temp, axes=range(3))

            # transform back to mn
            h_m = numpy.sum(h_xyz * m, axis=3)
            h_n = numpy.sum(h_xyz * n, axis=3)

        h.shape = (h.size,)
        h = numpy.concatenate((h_m.ravel(), h_n.ravel()), axis=None, out=h)     # ravel and merge
        h.shape = (h.size, 1)
        return h

    return operator


def hmn_2_exyz(
        k0: ArrayLike,
        G_matrix: ArrayLike,
        epsilon: fdfield_t,
        ) -> Callable[[NDArray[numpy.complex128]], cfdfield_t]:
    """
    Generate an operator which converts a vectorized spatial-frequency-space
     `h_mn` into an E-field distribution, i.e.

        ifft(conv(1/eps_k, ik x h_mn))

    The operator is a function that acts on a vector `h_mn` of size `2 * epsilon[0].size`.

    See the `meanas.fdfd.bloch` docstring for more information.

    Args:
        k0: Bloch wavevector, `[k0x, k0y, k0z]`.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        epsilon: Dielectric constant distribution for the simulation.
                 All fields are sampled at cell centers (i.e., NOT Yee-gridded)

    Returns:
        Function for converting `h_mn` into `E_xyz`
    """
    shape = epsilon[0].shape + (1,)
    epsilon = numpy.moveaxis(epsilon, 0, -1)

    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    def operator(h: NDArray[numpy.complex128]) -> cfdfield_t:
        hin_m, hin_n = (hi.reshape(shape) for hi in numpy.split(h, 2))
        d_xyz = (n * hin_m
               - m * hin_n) * k_mag         # noqa: E128

        # divide by epsilon
        return numpy.moveaxis(ifftn(d_xyz, axes=range(3)) / epsilon, 3, 0)

    return operator


def hmn_2_hxyz(
        k0: ArrayLike,
        G_matrix: ArrayLike,
        epsilon: fdfield_t
        ) -> Callable[[NDArray[numpy.complex128]], cfdfield_t]:
    """
    Generate an operator which converts a vectorized spatial-frequency-space
     `h_mn` into an H-field distribution, i.e.

        ifft(h_mn)

    The operator is a function that acts on a vector `h_mn` of size `2 * epsilon[0].size`.

    See the `meanas.fdfd.bloch` docstring for more information.

    Args:
        k0: Bloch wavevector, `[k0x, k0y, k0z]`.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        epsilon: Dielectric constant distribution for the simulation.
                 Only `epsilon[0].shape` is used.

    Returns:
        Function for converting `h_mn` into `H_xyz`
    """
    shape = epsilon[0].shape + (1,)
    _k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    def operator(h: NDArray[numpy.complex128]) -> cfdfield_t:
        hin_m, hin_n = (hi.reshape(shape) for hi in numpy.split(h, 2))
        h_xyz = (m * hin_m
               + n * hin_n)     # noqa: E128
        return numpy.array([ifftn(hi) for hi in numpy.moveaxis(h_xyz, 3, 0)])

    return operator


def inverse_maxwell_operator_approx(
        k0: ArrayLike,
        G_matrix: ArrayLike,
        epsilon: fdfield_t,
        mu: fdfield_t | None = None,
        ) -> Callable[[NDArray[numpy.complex128]], NDArray[numpy.complex128]]:
    """
    Generate an approximate inverse of the Maxwell operator,

        ik x conv(eps_k, ik x conv(mu_k, ___))

     which can be used to improve the speed of ARPACK in shift-invert mode.

    See the `meanas.fdfd.bloch` docstring for more information.

    Args:
        k0: Bloch wavevector, `[k0x, k0y, k0z]`.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        epsilon: Dielectric constant distribution for the simulation.
                 All fields are sampled at cell centers (i.e., NOT Yee-gridded)
        mu: Magnetic permability distribution for the simulation.
            Default None (1 everywhere).

    Returns:
        Function which applies the approximate inverse of the maxwell operator to `h_mn`.
    """
    shape = epsilon[0].shape + (1,)
    epsilon = numpy.moveaxis(epsilon, 0, -1)
    if mu is not None:
        mu = numpy.moveaxis(mu, 0, -1)

    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    def operator(h: NDArray[numpy.complex128]) -> NDArray[numpy.complex128]:
        """
        Approximate inverse Maxwell operator for Bloch eigenmode simulation.

        h is complex 2-field in (m, n) basis, vectorized

        Args:
            h: Raveled h_mn; size `2 * epsilon[0].size`.

        Returns:
            Raveled ik x conv(eps_k, ik x conv(mu_k, h_mn))
        """
        hin_m, hin_n = (hi.reshape(shape) for hi in numpy.split(h, 2))

        #{d,e,h}_xyz fields are complex 3-fields in (1/x, 1/y, 1/z) basis

        if mu is None:
            b_m, b_n = hin_m, hin_n
        else:
            # transform from mn to xyz
            h_xyz = (m * hin_m[:, :, :, None]
                   + n * hin_n[:, :, :, None])  # noqa: E128

            # multiply by mu
            temp = ifftn(h_xyz, axes=range(3))
            temp *= mu
            b_xyz = fftn(temp, axes=range(3))

            # transform back to mn
            b_m = numpy.sum(b_xyz * m, axis=3)
            b_n = numpy.sum(b_xyz * n, axis=3)

        # cross product and transform into xyz basis
        e_xyz = (n * b_m
               - m * b_n) / k_mag  # noqa: E128

        # multiply by epsilon
        temp = ifftn(e_xyz, axes=range(3))
        temp *= epsilon
        d_xyz = fftn(temp, axes=range(3))

        # cross product and transform into mn basis   crossinv_t2c
        h_m = numpy.sum(d_xyz * n, axis=3, keepdims=True) / +k_mag
        h_n = numpy.sum(d_xyz * m, axis=3, keepdims=True) / -k_mag

        h.shape = (h.size,)
        h = numpy.concatenate((h_m, h_n), axis=None, out=h)
        h.shape = (h.size, 1)
        return h

    return operator


def find_k(
        frequency: float,
        tolerance: float,
        direction: ArrayLike,
        G_matrix: ArrayLike,
        epsilon: fdfield_t,
        mu: fdfield_t | None = None,
        band: int = 0,
        k_bounds: tuple[float, float] = (0, 0.5),
        k_guess: float | None = None,
        solve_callback: Callable[..., None] | None = None,
        iter_callback: Callable[..., None] | None = None,
        v0: NDArray[numpy.complex128] | None = None,
        ) -> tuple[float, float, NDArray[numpy.complex128], NDArray[numpy.complex128]]:
    """
    Search for a bloch vector that has a given frequency.

    Args:
        frequency: Target frequency.
        tolerance: Target frequency tolerance.
        direction: k-vector direction to search along.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        epsilon: Dielectric constant distribution for the simulation.
                 All fields are sampled at cell centers (i.e., NOT Yee-gridded)
        mu: Magnetic permability distribution for the simulation.
            Default None (1 everywhere).
        band: Which band to search in. Default 0 (lowest frequency).
        k_bounds: Minimum and maximum values for k. Default (0, 0.5).
        k_guess: Initial value for k.
        solve_callback: TODO
        iter_callback: TODO

    Returns:
        `(k, actual_frequency, eigenvalues, eigenvectors)`
        The found k-vector and its frequency, along with all eigenvalues and eigenvectors.
    """
    direction = numpy.array(direction) / norm(direction)

    k_bounds = tuple(sorted(k_bounds))    # type: ignore    # we know the length already...
    assert len(k_bounds) == 2

    if k_guess is None:
        k_guess = sum(k_bounds) / 2

    n = None
    v = v0

    def get_f(k0_mag: float, band: int = 0) -> float:
        nonlocal n, v
        k0 = direction * k0_mag                         # type: ignore
        n, v = eigsolve(band + 1, k0, G_matrix=G_matrix, epsilon=epsilon, mu=mu, y0=v, callback=iter_callback)
        f = numpy.sqrt(numpy.abs(numpy.real(n[band])))
        if solve_callback:
            solve_callback(k0_mag, n, v, f)
        return f

    res = scipy.optimize.minimize_scalar(
        lambda x: abs(get_f(x, band) - frequency),
        k_guess,
        method='Bounded',
        bounds=k_bounds,
        options={'xatol': abs(tolerance)},
        )

    assert n is not None
    assert v is not None
    return float(res.x * direction), float(res.fun + frequency), n, v


def eigsolve(
        num_modes: int,
        k0: ArrayLike,
        G_matrix: ArrayLike,
        epsilon: fdfield_t,
        mu: fdfield_t | None = None,
        tolerance: float = 1e-7,
        max_iters: int = 10000,
        reset_iters: int = 100,
        y0: ArrayLike | None = None,
        callback: Callable[..., None] | None = None,
        ) -> tuple[NDArray[numpy.complex128], NDArray[numpy.complex128]]:
    """
    Find the first (lowest-frequency) num_modes eigenmodes with Bloch wavevector
     k0 of the specified structure.

    Args:
        k0: Bloch wavevector, `[k0x, k0y, k0z]`.
        G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
        epsilon: Dielectric constant distribution for the simulation.
                 All fields are sampled at cell centers (i.e., NOT Yee-gridded)
        mu: Magnetic permability distribution for the simulation.
            Default `None` (1 everywhere).
        tolerance: Solver stops when fractional change in the objective
                   `trace(Z.H @ A @ Z @ inv(Z Z.H))` is smaller than the tolerance
        max_iters: TODO
        reset_iters: TODO
        callback: TODO
        y0: TODO, initial guess

    Returns:
        `(eigenvalues, eigenvectors)` where `eigenvalues[i]` corresponds to the
        vector `eigenvectors[i, :]`
    """
    k0 = numpy.asarray(k0)

    h_size = 2 * epsilon[0].size

    kmag = norm(G_matrix @ k0)

    #
    # Generate the operators
    #
    mop = maxwell_operator(k0=k0, G_matrix=G_matrix, epsilon=epsilon, mu=mu)
    imop = inverse_maxwell_operator_approx(k0=k0, G_matrix=G_matrix, epsilon=epsilon, mu=mu)

    scipy_op = spalg.LinearOperator(dtype=complex, shape=(h_size, h_size), matvec=mop)
    scipy_iop = spalg.LinearOperator(dtype=complex, shape=(h_size, h_size), matvec=imop)

    y_shape = (h_size, num_modes)

    prev_E = 0.0
    d_scale = 1.0
    prev_traceGtKG = 0.0
    prev_theta = 0.5
    D = numpy.zeros(shape=y_shape, dtype=complex)

    rng = numpy.random.default_rng()
    Z: NDArray[numpy.complex128]
    if y0 is None:
        Z = rng.random(y_shape) + 1j * rng.random(y_shape)
    else:
        Z = numpy.asarray(y0).T

    while True:
        Z *= num_modes / norm(Z)
        ZtZ = Z.conj().T @ Z
        try:
            U = numpy.linalg.inv(ZtZ)
        except numpy.linalg.LinAlgError:
            Z = rng.random(y_shape) + 1j * rng.random(y_shape)
            continue

        trace_U = real(trace(U))
        if trace_U > 1e8 * num_modes:
            Z = Z @ scipy.linalg.sqrtm(U).astype(numpy.complex128).conj().T
            prev_traceGtKG = 0
            continue
        break

    Zt = numpy.empty(Z.shape[::-1], dtype=numpy.complex128)
    AZ = numpy.empty(Z.shape, dtype=numpy.complex128)

    for i in range(max_iters):
        Zt = numpy.conj(Z.T, out=Zt)
        ZtZ = Zt @ Z
        U = numpy.linalg.inv(ZtZ)
        AZ = scipy_op @ Z.copy()
        ZtAZ = Zt @ AZ
        ZtAZU = ZtAZ @ U
        E_signed = real(trace(ZtAZU))
        sgn = numpy.sign(E_signed)
        E = numpy.abs(E_signed)

        # G = AZU projected onto the space orthonormal to Z via (1 - ZUZt)
        G = (AZ @ U - Z @ U @ ZtAZU) * sgn

        if i > 0 and abs(E - prev_E) < tolerance * 0.5 * (E + prev_E + 1e-7):
            logger.info(
                'Optimization succeded: '
                f'[change in trace] {abs(E - prev_E)} - 5e-8 '
                f'< {tolerance} [tolerance] * {(E + prev_E) / 2} [value of trace]'
                )
            break

        KG = scipy_iop @ G          # Preconditioned steepest descent direction
        traceGtKG = _rtrace_AtB(G, KG)

        if prev_traceGtKG == 0 or i % reset_iters == 0:
            logger.info('CG reset')
            gamma = 0.0
        else:
            gamma = traceGtKG / prev_traceGtKG
        prev_traceGtKG = traceGtKG

        D = gamma / d_scale * D + KG
        d_scale = num_modes / norm(D)
        D *= d_scale

        # Now know the direction (D), but need to find how far to go (alpha)
        # We are still minimizing E = tr((Z + alpha D)t A (Z + alpha D) U')
        #         = tr(ZtAZU' + alpha DtAZU' + alpha ZtADU' + alpha**2 DtADU')
        #         = tr((ZtAZ + 2 alpha sym(DtAZ) + alpha**2 DtAD) U')
        #         = tr(R U')
        #         = tr(R Qi) = tr(R inv(Q))

        # where
        #   R = ZtAZ + 2 alpha sym(DtAZ) + alpha**2 DtAD
        #
        #   Q = (Z + alpha D)t (Z + alpha D)
        #     = inv(ZtZ + alpha ZtD + alpha DtZ + alpha**2 DtD)
        #
        #   Qi = inv(Q) = U'

        AD = scipy_op @ D.copy()
        DtD = D.conj().T @ D
        DtAD = D.conj().T @ AD

        symZtD = _symmetrize(Zt @ D)
        symZtAD = _symmetrize(Zt @ AD)

        Qi_memo: list[float | None] = [None, None]

        def Qi_func(theta: float, Qi_memo=Qi_memo, ZtZ=ZtZ, DtD=DtD, symZtD=symZtD) -> float:   # noqa: ANN001
            if Qi_memo[0] == theta:
                return cast(float, Qi_memo[1])

            c = numpy.cos(theta)
            s = numpy.sin(theta)
            Q = c * c * ZtZ + s * s * DtD + 2 * s * c * symZtD
            try:
                Qi = numpy.linalg.inv(Q)
            except numpy.linalg.LinAlgError as err:
                logger.info('taylor Qi')
                # if c or s small, taylor expand
                if c < 1e-4 * s and c != 0:
                    DtDi = numpy.linalg.inv(DtD)
                    Qi = DtDi / (s * s) - 2 * c / (s * s * s) * (DtDi @ (DtDi @ symZtD).conj().T)
                elif s < 1e-4 * c and s != 0:
                    ZtZi = numpy.linalg.inv(ZtZ)
                    Qi = ZtZi / (c * c) - 2 * s / (c * c * c) * (ZtZi @ (ZtZi @ symZtD).conj().T)
                else:
                    raise Exception('Inexplicable singularity in trace_func') from err
            Qi_memo[0] = theta
            Qi_memo[1] = cast(float, Qi)
            return cast(float, Qi)

        def trace_func(theta: float, ZtAZ=ZtAZ, DtAD=DtAD, symZtAD=symZtAD) -> float:           # noqa: ANN001
            c = numpy.cos(theta)
            s = numpy.sin(theta)
            Qi = Qi_func(theta)
            R = c * c * ZtAZ + s * s * DtAD + 2 * s * c * symZtAD
            trace = _rtrace_AtB(R, Qi)
            return numpy.abs(trace)

        if False:
            def trace_deriv(theta, sgn: int = sgn, ZtAZ=ZtAZ, DtAD=DtAD, symZtD=symZtD, symZtAD=symZtAD, ZtZ=ZtZ, DtD=DtD):     # noqa: ANN001
                Qi = Qi_func(theta)
                c2 = numpy.cos(2 * theta)
                s2 = numpy.sin(2 * theta)
                F = -0.5 * s2 * (ZtAZ - DtAD) + c2 * symZtAD
                trace_deriv = _rtrace_AtB(Qi, F)

                G = Qi @ F.conj().T @ Qi.conj().T
                H = -0.5 * s2 * (ZtZ - DtD) + c2 * symZtD
                trace_deriv -= _rtrace_AtB(G, H)

                trace_deriv *= 2
                return trace_deriv * sgn

            U_sZtD = U @ symZtD

            dE = 2.0 * (_rtrace_AtB(U, symZtAD)
                        - _rtrace_AtB(ZtAZU, U_sZtD))

            d2E = 2 * (_rtrace_AtB(U, DtAD)
                       - _rtrace_AtB(ZtAZU, U @ (DtD - 4 * symZtD @ U_sZtD))
                       - 4 * _rtrace_AtB(U, symZtAD @ U_sZtD))

            # Newton-Raphson to find a root of the first derivative:
            theta = -dE / d2E

            if d2E < 0 or abs(theta) >= pi:
                theta = -abs(prev_theta) * numpy.sign(dE)

            # theta, new_E, new_dE = linmin(theta, E, dE, 0.1, min(tolerance, 1e-6), 1e-14, 0, -numpy.sign(dE) * K_PI, trace_func)
            theta, n, _, new_E, _, _new_dE = scipy.optimize.line_search(
                trace_func,
                trace_deriv,
                xk=theta,
                pk=numpy.ones((1, 1)),
                gfk=dE,
                old_fval=E,
                c1=min(tolerance, 1e-6),
                c2=0.1,
                amax=pi,
                )

        result = scipy.optimize.minimize_scalar(trace_func, bounds=(0, pi), tol=tolerance)
        new_E = result.fun
        theta = result.x

        improvement = numpy.abs(E - new_E) * 2 / numpy.abs(E + new_E)
        logger.info(f'linmin improvement {improvement}')
        Z *= numpy.cos(theta)
        Z += D * numpy.sin(theta)

        prev_theta = theta
        prev_E = E

        if callback:
            callback()

    #
    # Recover eigenvectors from Z
    #
    U = numpy.linalg.inv(ZtZ)
    Y = Z @ scipy.linalg.sqrtm(U).astype(numpy.complex128)
    W = Y.conj().T @ (scipy_op @ Y.copy())

    eigvals, W_eigvecs = numpy.linalg.eig(W)
    eigvecs = Y @ W_eigvecs

    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        n = eigvals[i]
        v /= norm(v)
        Av = (scipy_op @ v.copy())[:, 0]
        eigness = norm(Av - (v.conj() @ Av) * v)
        f = numpy.sqrt(-numpy.real(n))
        df = numpy.sqrt(-numpy.real(n) + eigness)
        neff_err = kmag * (1 / df - 1 / f)
        logger.info(f'eigness {i}: {eigness}\n neff_err: {neff_err}')

    order = numpy.argsort(numpy.abs(eigvals))
    return eigvals[order], eigvecs.T[order]


'''
def linmin(x_guess, f0, df0, x_max, f_tol=0.1, df_tol=min(tolerance, 1e-6), x_tol=1e-14, x_min=0, linmin_func):
    if df0 > 0:
        x0, f0, df0 = linmin(-x_guess, f0, -df0, -x_max, f_tol, df_tol, x_tol, -x_min, lambda q, dq: -linmin_func(q, dq))
        return -x0, f0, -df0
    elif df0 == 0:
        return 0, f0, df0
    else:
        x = x_guess
        fx = f0
        dfx = df0

        isave = numpy.zeros((2,), numpy.intc)
        dsave = numpy.zeros((13,), float)

        x, fx, dfx, task = minpack2.dsrch(x, fx, dfx, f_tol, df_tol, x_tol, task,
                                          x_min, x_max, isave, dsave)
        for i in range(int(1e6)):
            if task != 'F':
                logging.info(f'search converged in {i} iterations')
                break
            fx = f(x, dfx)
            x, fx, dfx, task = minpack2.dsrch(x, fx, dfx, f_tol, df_tol, x_tol, task,
                                              x_min, x_max, isave, dsave)

        return x, fx, dfx
'''

def _rtrace_AtB(
        A: NDArray[numpy.complex128],
        B: NDArray[numpy.complex128] | float,
        ) -> float:
    return real(numpy.sum(A.conj() * B))

def _symmetrize(A: NDArray[numpy.complex128]) -> NDArray[numpy.complex128]:
    return (A + A.conj().T) * 0.5



def inner_product(eL, hL, eR, hR) -> complex:
    # assumes x-axis propagation

    assert numpy.array_equal(eR.shape, hR.shape)
    assert numpy.array_equal(eL.shape, hL.shape)
    assert numpy.array_equal(eR.shape, eL.shape)

    # Cross product, times 2 since it's <p | n>, then divide by 4. # TODO might want to abs() this?
    norm2R = (eR[1] * hR[2] - eR[2] * hR[1]).sum() / 2
    norm2L = (eL[1] * hL[2] - eL[2] * hL[1]).sum() / 2

    # eRxhR_x = numpy.cross(eR.reshape(3, -1), hR.reshape(3, -1), axis=0).reshape(eR.shape)[0] / normR
    # logger.info(f'power {eRxhR_x.sum() / 2})

    eR /= numpy.sqrt(norm2R)
    hR /= numpy.sqrt(norm2R)
    eL /= numpy.sqrt(norm2L)
    hL /= numpy.sqrt(norm2L)

    # (eR x hL)[0] and (eL x hR)[0]
    eRxhL_x = eR[1] * hL[2] - eR[2] - hL[1]
    eLxhR_x = eL[1] * hR[2] - eL[2] - hR[1]

    #return 1j *  (eRxhL_x - eLxhR_x).sum() / numpy.sqrt(norm2R * norm2L)
    #return (eRxhL_x.sum() - eLxhR_x.sum()) / numpy.sqrt(norm2R * norm2L)
    return eRxhL_x.sum() - eLxhR_x.sum()


def trq(eI, hI, eO, hO) -> tuple[complex, complex]:
    pp = inner_product(eO,  hO, eI,  hI)
    pn = inner_product(eO,  hO, eI, -hI)
    np = inner_product(eO, -hO, eI,  hI)
    nn = inner_product(eO, -hO, eI, -hI)

    assert pp == -nn
    assert pn == -np

    logger.info(f'''
        {pp=:4g} {pn=:4g}
        {nn=:4g} {np=:4g}
            {nn * pp / pn=:4g}    {-np=:4g}
        ''')

    r = -pp / pn    # -<Pp|Bp>/<Pn/Bp> = -(-pp) / (-pn)
    t = (np - nn * pp / pn) / 4

    return t, r
