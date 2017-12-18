'''
Bloch eigenmode solver/operators

This module contains functions for generating and solving the
 3D Bloch eigenproblem. The approach is to transform the problem
 into the (spatial) fourier domain, transforming the equation
   1/mu * curl(1/eps * curl(H)) = (w/c)^2 H
 into
   conv(1/mu_k, ik x conv(1/eps_k, ik x H_k)) = (w/c)^2 H_k
 where:
  - the _k subscript denotes a 3D fourier transformed field
  - each component of H_k corresponds to a plane wave with wavevector k
  - x is the cross product
  - conv denotes convolution

 Since k and H are orthogonal for each plane wave, we can use each
 k to create an orthogonal basis (k, m, n), with k x m = n, and
 |m| = |n| = 1. The cross products are then simplified with

 k @ h = kx hx + ky hy + kz hz = 0 = hk
 h = hk + hm + hn = hm + hn
 k = kk + km + kn = kk = |k|

 k x h = (ky hz - kz hy,
          kz hx - kx hz,
          kx hy - ky hx)
       = ((k x h) @ k, (k x h) @ m, (k x h) @ n)_kmn
       = (0, (m x k) @ h, (n x k) @ h)_kmn         # triple product ordering
       = (0, kk (-n @ h), kk (m @ h))_kmn          # (m x k) = -|k| n, etc.
       = |k| (0, -h @ n, h @ m)_kmn

 k x h = (km hn - kn hm,
          kn hk - kk hn,
          kk hm - km hk)_kmn
       = (0, -kk hn, kk hm)_kmn
       = (-kk hn)(mx, my, mz) + (kk hm)(nx, ny, nz)
       = |k| (hm * (nx, ny, nz) - hn * (mx, my, mz))

 where h is shorthand for H_k, (...)_kmn deontes the (k, m, n) basis,
 and e.g. hm is the component of h in the m direction.

 We can also simplify conv(X_k, Y_k) as fftn(X * ifftn(Y_k)).

 Using these results and storing H_k as h = (hm, hn), we have
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

'''

from typing import List, Tuple, Callable, Dict
import logging
import numpy
from numpy.fft import fftn, ifftn, fftfreq
import scipy
import scipy.optimize
from scipy.linalg import norm
import scipy.sparse.linalg as spalg

from .eigensolvers import rayleigh_quotient_iteration
from . import field_t

logger = logging.getLogger(__name__)


def generate_kmn(k0: numpy.ndarray,
                 G_matrix: numpy.ndarray,
                 shape: numpy.ndarray
                 ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Generate a (k, m, n) orthogonal basis for each k-vector in the simulation grid.

    :param k0: [k0x, k0y, k0z], Bloch wavevector, in G basis.
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param shape: [nx, ny, nz] shape of the simulation grid.
    :return: (|k|, m, n) where |k| has shape tuple(shape) + (1,)
            and m, n have shape tuple(shape) + (3,).
            All are given in the xyz basis (e.g. |k|[0,0,0] = norm(G_matrix @ k0)).
    """
    k0 = numpy.array(k0)

    Gi_grids = numpy.meshgrid(*(fftfreq(n, 1/n) for n in shape[:3]), indexing='ij')
    Gi = numpy.stack(Gi_grids, axis=3)

    k_G = k0[None, None, None, :] - Gi
    k_xyz = numpy.rollaxis(G_matrix @ numpy.rollaxis(k_G, 3, 2), 3, 2)

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


def maxwell_operator(k0: numpy.ndarray,
                     G_matrix: numpy.ndarray,
                     epsilon: field_t,
                     mu: field_t = None
                     ) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """
    Generate the Maxwell operator
        conv(1/mu_k, ik x conv(1/eps_k, ik x ___))
    which is the spatial-frequency-space representation of
        1/mu * curl(1/eps * curl(___))

    The operator is a function that acts on a vector h_mn of size (2 * epsilon[0].size)

    See the module-level docstring for more information.

    :param k0: Bloch wavevector, [k0x, k0y, k0z].
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param epsilon: Dielectric constant distribution for the simulation.
        All fields are sampled at cell centers (i.e., NOT Yee-gridded)
    :param mu: Magnetic permability distribution for the simulation.
        Default None (1 everywhere).
    :return: Function which applies the maxwell operator to h_mn.
    """

    shape = epsilon[0].shape + (1,)
    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    epsilon = numpy.stack(epsilon, 3)
    if mu is not None:
        mu = numpy.stack(mu, 3)

    def operator(h: numpy.ndarray):
        """
        Maxwell operator for Bloch eigenmode simulation.

        h is complex 2-field in (m, n) basis, vectorized

        :param h: Raveled h_mn; size (2 * epsilon[0].size).
        :return: Raveled conv(1/mu_k, ik x conv(1/eps_k, ik x h_mn)).
        """
        hin_m, hin_n = [hi.reshape(shape) for hi in numpy.split(h, 2)]

        #{d,e,h}_xyz fields are complex 3-fields in (1/x, 1/y, 1/z) basis

        # cross product and transform into xyz basis
        d_xyz = (n * hin_m -
                 m * hin_n) * k_mag

        # divide by epsilon
        e_xyz = fftn(ifftn(d_xyz, axes=range(3)) / epsilon, axes=range(3))

        # cross product and transform into mn basis
        b_m = numpy.sum(e_xyz * n, axis=3)[:, :, :, None] * -k_mag
        b_n = numpy.sum(e_xyz * m, axis=3)[:, :, :, None] * +k_mag

        if mu is None:
            h_m, h_n = b_m, b_n
        else:
            # transform from mn to xyz
            b_xyz = (m * b_m[:, :, :, None] +
                     n * b_n[:, :, :, None])

            # divide by mu
            h_xyz = fftn(ifftn(b_xyz, axes=range(3)) / mu, axes=range(3))

            # transform back to mn
            h_m = numpy.sum(h_xyz * m, axis=3)
            h_n = numpy.sum(h_xyz * n, axis=3)
        return numpy.hstack((h_m.ravel(), h_n.ravel()))

    return operator


def hmn_2_exyz(k0: numpy.ndarray,
               G_matrix: numpy.ndarray,
               epsilon: field_t,
               ) -> Callable[[numpy.ndarray], field_t]:
    """
    Generate an operator which converts a vectorized spatial-frequency-space
     h_mn into an E-field distribution, i.e.
        ifft(conv(1/eps_k, ik x h_mn))

    The operator is a function that acts on a vector h_mn of size (2 * epsilon[0].size)

    See the module-level docstring for more information.

    :param k0: Bloch wavevector, [k0x, k0y, k0z].
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param epsilon: Dielectric constant distribution for the simulation.
        All fields are sampled at cell centers (i.e., NOT Yee-gridded)
    :return: Function for converting h_mn into E_xyz
    """
    shape = epsilon[0].shape + (1,)
    epsilon = numpy.stack(epsilon, 3)

    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    def operator(h: numpy.ndarray) -> field_t:
        hin_m, hin_n = [hi.reshape(shape) for hi in numpy.split(h, 2)]
        d_xyz = (n * hin_m -
                 m * hin_n) * k_mag

        # divide by epsilon
        return [ei for ei in numpy.rollaxis(ifftn(d_xyz, axes=range(3)) / epsilon, 3)]

    return operator


def hmn_2_hxyz(k0: numpy.ndarray,
               G_matrix: numpy.ndarray,
               epsilon: field_t
               ) -> Callable[[numpy.ndarray], field_t]:
    """
    Generate an operator which converts a vectorized spatial-frequency-space
     h_mn into an H-field distribution, i.e.
        ifft(h_mn)

    The operator is a function that acts on a vector h_mn of size (2 * epsilon[0].size)

    See the module-level docstring for more information.

    :param k0: Bloch wavevector, [k0x, k0y, k0z].
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param epsilon: Dielectric constant distribution for the simulation.
        Only epsilon[0].shape is used.
    :return: Function for converting h_mn into H_xyz
    """
    shape = epsilon[0].shape + (1,)
    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    def operator(h: numpy.ndarray):
        hin_m, hin_n = [hi.reshape(shape) for hi in numpy.split(h, 2)]
        h_xyz = (m * hin_m +
                 n * hin_n)
        return [ifftn(hi) for hi in numpy.rollaxis(h_xyz, 3)]

    return operator


def inverse_maxwell_operator_approx(k0: numpy.ndarray,
                                    G_matrix: numpy.ndarray,
                                    epsilon: field_t,
                                    mu: field_t = None
                                    ) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """
    Generate an approximate inverse of the Maxwell operator,
        ik x conv(eps_k, ik x conv(mu_k, ___))
     which can be used to improve the speed of ARPACK in shift-invert mode.

    See the module-level docstring for more information.

    :param k0: Bloch wavevector, [k0x, k0y, k0z].
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param epsilon: Dielectric constant distribution for the simulation.
        All fields are sampled at cell centers (i.e., NOT Yee-gridded)
    :param mu: Magnetic permability distribution for the simulation.
        Default None (1 everywhere).
    :return: Function which applies the approximate inverse of the maxwell operator to h_mn.
    """
    shape = epsilon[0].shape + (1,)
    epsilon = numpy.stack(epsilon, 3)

    k_mag, m, n = generate_kmn(k0, G_matrix, shape)

    if mu is not None:
        mu = numpy.stack(mu, 3)

    def operator(h: numpy.ndarray):
        """
        Approximate inverse Maxwell operator for Bloch eigenmode simulation.

        h is complex 2-field in (m, n) basis, vectorized

        :param h: Raveled h_mn; size (2 * epsilon[0].size).
        :return: Raveled ik x conv(eps_k, ik x conv(mu_k, h_mn))
        """
        hin_m, hin_n = [hi.reshape(shape) for hi in numpy.split(h, 2)]

        #{d,e,h}_xyz fields are complex 3-fields in (1/x, 1/y, 1/z) basis

        if mu is None:
            b_m, b_n = hin_m, hin_n
        else:
            # transform from mn to xyz
            h_xyz = (m * hin_m[:, :, :, None] +
                     n * hin_n[:, :, :, None])

            # multiply by mu
            b_xyz = fftn(ifftn(h_xyz, axes=range(3)) * mu, axes=range(3))

            # transform back to mn
            b_m = numpy.sum(b_xyz * m, axis=3)
            b_n = numpy.sum(b_xyz * n, axis=3)

        # cross product and transform into xyz basis
        e_xyz = (n * b_m -
                 m * b_n) / k_mag

        # multiply by epsilon
        d_xyz = fftn(ifftn(e_xyz, axes=range(3)) * epsilon, axes=range(3))

        # cross product and transform into mn basis   crossinv_t2c
        h_m = numpy.sum(e_xyz * n, axis=3)[:, :, :, None] / +k_mag
        h_n = numpy.sum(e_xyz * m, axis=3)[:, :, :, None] / -k_mag

        return numpy.hstack((h_m.ravel(), h_n.ravel()))

    return operator


def eigsolve(num_modes: int,
             k0: numpy.ndarray,
             G_matrix: numpy.ndarray,
             epsilon: field_t,
             mu: field_t = None,
             tolerance = 1e-8,
             ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Find the first (lowest-frequency) num_modes eigenmodes with Bloch wavevector
     k0 of the specified structure.

    :param k0: Bloch wavevector, [k0x, k0y, k0z].
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param epsilon: Dielectric constant distribution for the simulation.
        All fields are sampled at cell centers (i.e., NOT Yee-gridded)
    :param mu: Magnetic permability distribution for the simulation.
        Default None (1 everywhere).
    :return: (eigenvalues, eigenvectors) where eigenvalues[i] corresponds to the
        vector eigenvectors[i, :]
    """
    h_size = 2 * epsilon[0].size

    '''
    Generate the operators
    '''
    mop = maxwell_operator(k0=k0, G_matrix=G_matrix, epsilon=epsilon, mu=mu)
    imop = inverse_maxwell_operator_approx(k0=k0, G_matrix=G_matrix, epsilon=epsilon, mu=mu)

    scipy_op = spalg.LinearOperator(dtype=complex, shape=(h_size, h_size), matvec=mop)
    scipy_iop = spalg.LinearOperator(dtype=complex, shape=(h_size, h_size), matvec=imop)

    y_shape = (h_size, num_modes)

    def rayleigh_quotient(Z: numpy.ndarray, approx_grad: bool = True):
        """
        Absolute value of the block Rayleigh quotient, and the associated gradient.

        See Johnson and Joannopoulos, Opt. Expr. 8, 3 (2001) for details (full
         citation in module docstring).

        ===

        Notes on my understanding of the procedure:

        Minimize f(Y) = |trace((Y.H @ A @ Y)|, making use of Y = Z @ inv(Z.H @ Z)^(1/2)
         (a polar orthogonalization of Y). This gives f(Z) = |trace(Z.H @ A @ Z @ U)|,
         where U = inv(Z.H @ Z). We minimize the absolute value to find the eigenvalues
         with smallest magnitude.

        The gradient is P @ (A @ Z @ U), where P = (1 - Z @ U @ Z.H) is a projection
         onto the space orthonormal to Z. If approx_grad is True, the approximate
         inverse of the maxwell operator is used to precondition the gradient.
        """
        z = Z.reshape(y_shape)
        U = numpy.linalg.inv(z.conj().T @ z)
        zU = z @ U
        AzU = scipy_op @ zU
        zTAzU = z.conj().T @ AzU
        f = numpy.real(numpy.trace(zTAzU))
        if approx_grad:
            df_dy = scipy_iop @ (AzU - zU @ zTAzU)
        else:
            df_dy = (AzU - zU @ zTAzU)
        return numpy.abs(f), numpy.sign(f) * numpy.real(df_dy).ravel()

    '''
    Use the conjugate gradient method and the approximate gradient calculation to
     quickly find approximate eigenvectors.
    '''
    result = scipy.optimize.minimize(rayleigh_quotient,
                                     numpy.random.rand(*y_shape),
                                     jac=True,
                                     method='CG',
                                     tol=1e-5,
                                     options={'maxiter': 30, 'disp':True})

    result = scipy.optimize.minimize(lambda y: rayleigh_quotient(y, False),
                                     result.x,
                                     jac=True,
                                     method='CG',
                                     tol=1e-13,
                                     options={'maxiter': 100, 'disp':True})

    z = result.x.reshape(y_shape)

    '''
    Recover eigenvectors from Z
    '''
    U = numpy.linalg.inv(z.conj().T @ z)
    y = z @ scipy.linalg.sqrtm(U)
    w = y.conj().T @ (scipy_op @ y)

    eigvals, w_eigvecs = numpy.linalg.eig(w)
    eigvecs = y @ w_eigvecs

    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        n = eigvals[i]
        v /= norm(v)
        logger.info('eigness {}: {}'.format(i, norm(scipy_op @ v - (v.conj() @ (scipy_op @ v)) * v )))

    ev2 = eigvecs.copy()
    for i in range(len(eigvals)):
        logger.info('Refining eigenvector {}'.format(i))
        eigvals[i], ev2[:, i] = rayleigh_quotient_iteration(scipy_op,
                                                            guess_vector=eigvecs[:, i],
                                                            iterations=40,
                                                            tolerance=tolerance * numpy.real(numpy.sqrt(eigvals[i])) * 2,
                                                            solver = lambda A, b: spalg.bicgstab(A, b, maxiter=200)[0])
    eigvecs = ev2
    order = numpy.argsort(numpy.abs(eigvals))

    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        n = eigvals[i]
        v /= norm(v)
        logger.info('eigness {}: {}'.format(i, norm(scipy_op @ v - (v.conj() @ (scipy_op @ v)) * v )))

    return eigvals[order], eigvecs.T[order]


def find_k(frequency: float,
           tolerance: float,
           direction: numpy.ndarray,
           G_matrix: numpy.ndarray,
           epsilon: field_t,
           mu: field_t = None,
           band: int = 0
           ) -> Tuple[numpy.ndarray, float]:
    """
    Search for a bloch vector that has a given frequency.

    :param frequency: Target frequency.
    :param tolerance: Target frequency tolerance.
    :param direction: k-vector direction to search along.
    :param G_matrix: 3x3 matrix, with reciprocal lattice vectors as columns.
    :param epsilon: Dielectric constant distribution for the simulation.
        All fields are sampled at cell centers (i.e., NOT Yee-gridded)
    :param mu: Magnetic permability distribution for the simulation.
        Default None (1 everywhere).
    :param band: Which band to search in. Default 0 (lowest frequency).
    return: (k, actual_frequency) The found k-vector and its frequency
    """

    direction = numpy.array(direction) / norm(direction)

    def get_f(k0_mag: float, band: int = 0):
        k0 = direction * k0_mag
        n, _v = eigsolve(band + 1, k0, G_matrix=G_matrix, epsilon=epsilon)
        f = numpy.sqrt(numpy.abs(numpy.real(n[band])))
        return f

    res = scipy.optimize.minimize_scalar(lambda x: abs(get_f(x, band) - frequency), 0.25,
                                         method='Bounded', bounds=(0, 0.5),
                                         options={'xatol': abs(tolerance)})
    return res.x * direction, res.fun + frequency


