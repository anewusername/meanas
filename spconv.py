import scipy
import numpy
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import pinv
from numpy import sqrt, real, abs, pi


def diag(twod):
    # numpy.diag() but makes a stack of diagonal matrices
    return numpy.einsum('ij,jk->ijk', twod, numpy.eye(twod.shape[-1], dtype=twod.dtype))


def s2z(s, zref):
    # G0_inv @ inv(I - S) @ (S Z0 + Z0*) @ G0
    #  Where Z0 is diag(zref) and G0 = diag(1/sqrt(abs(real(zref))))
    nf = s.shape[-1]
    I = numpy.eye(nf)[None, :, :]
    zref = numpy.array(zref, copy=False)
    gref = 1 / sqrt(abs(zref.real))
    z = diag(1 / gref) @ pinv(I - s) @ ( S @ diag(zref) + diag(zref).conj()) @ diag(gref)
    return z


def change_of_zref(
        s,  # (nf, np, np)
        zref_old,  # (nf, np)
        zref_new,  # (nf, np)
        ):
    # Change-of-Z0 to Z0'
    # S' = inv(A) @ (S - rho*) @ inv(I - rho @ S) @ A*
    #  A = inv(G0') @ G0 @ inv(I - rho*)   (diagonal)
    #  rho = (Z0' - Z0) @ inv(Z0' + Z0)    (diagonal)

    I = numpy.zeros_like(SL)
    numpy.einsum('...jj->...j', I)[...] = 1

    zref_old = numpy.array(zref_old, copy=False)
    zref_new = numpy.array(zref_new, copy=False)

    g_old = 1 / sqrt(abs(zref_old.real))
    r_new = sqrt(abs(zref_new.real))

    rhov = (zref_new - zref_old) / (zref_new + zref_old)
    av = r_new * g_old / (1 - rhov.conj())

    s_new = diag(1 / av) @ (s - diag(rhov.conj())) @ pinv(I[None, :] - diag(rhov) @ S) @ diag(av.conj())
    return s_new


def embedding(
        See,  # (nf, ne, ne)
        Sei,  # (nf, ne, ni)
        Sie,  # (nf, ni, ne)
        Sii,  # (nf, ni, ni)
        SL,   # (nf, ni, ni)
        ):
    # Reveyrand, doi:10.1109/INMMIC.2018.8430023

    I = numpy.zeros_like(SL)
    numpy.einsum('...jj->...j', I)[...] = 1

    S_tot = See + Sei @ pinv(I - SL @ Sii) @ SL @ Sie
    return S_tot


def deembedding(
        Sei,  # (nf, ne, ni)
        Sie,  # (nf, ni, ne)
        Sext, # (nf, ne, ne)
        See,  # (nf, ne, ne)
        Si,   # (nf, ni, ni)
        ):
    # Reveyrand, doi:10.1109/INMMIC.2018.8430023
    # Requires balanced number of ports, similar to VNA calibration
    Sei_inv = pinv(Sei)
    Sdif = Sext - See
    return Sei_inv @ Sdif @ pinv(Sie + Sii @ Sei_inv @ Sdif)


def thru_with_Zref_change(
        zref0,  # (nf,)
        zref1,  # (nf,)
        ):
    s = numpy.empty(tuple(zref0.shape) + (2, 2), dtype=complex)
    s[..., 0, 0] = zref1 - zref0
    s[..., 0, 1] = 2 * sqrt(zref0 * zref1)
    s[..., 1, 0] = s[..., 0, 1]
    s[..., 1, 1] = -s[..., 0, 0]

    s /= zref0 + zref1
    return s


def propagation_matrix(mode_neffs: ArrayLike, wavelength: float, distance: float):
    eigenv = numpy.array(mode_neffs, copy=False) * 2 * pi / wavelength
    prop_diag = numpy.diag(numpy.exp(distance * 1j * numpy.hstack((eigenv, eigenv))))
    prop_matrix = numpy.roll(prop_diag, len(eigenv), axis=0)
    return prop_matrix


def connect_s(
        A: NDArray[numpy.complex128],
        k: int,
        B: NDArray[numpy.complex128],
        l: int,
        ) -> NDArray[numpy.complex128]:
    """
    TODO
    freq x ...  x n x n

    Based on skrf implementation

    Connect two n-port networks' s-matrices together.

    Specifically, connect port `k` on network `A` to port `l` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2); first
    (A.rank - 1) ports are from `A`, remainder are from B.

    Assumes same reference impedance for both `k` and `l`; may need to
    connect an "impedance mismatch" thru element first!

    Args:
        A: S-parameter matrix of `A`, shape is fxnxn
        k: port index on `A` (port indices start from 0)
        B: S-parameter matrix of `B`, shape is fxnxn
        l: port index on `B`

    Returns:
        new S-parameter matrix
    """
    if k > A.shape[-1] - 1 or l > B.shape[-1] - 1:
        raise ValueError("port indices are out of range")

    #C = scipy.sparse.block_diag((A, B), dtype=complex)
    #return innerconnect_s(C, k, A.shape[0] + l)

    nA = A.shape[-1]
    nB = B.shape[-1]
    nC = nA + nB - 2
    assert numpy.array_equal(A.shape[:-2], B.shape[:-2])

    ll = slice(l, l + 1)
    kk = slice(k, k + 1)

    denom = 1 - A[..., kk, kk] * B[..., ll, ll]
    Anew = A + A[..., kk, :] * B[..., ll, ll] * A[..., :, kk] / denom
    Bnew = A[..., kk, :] * B[..., :, ll] / denom
    Anew = numpy.delete(Anew, (k,), 1)
    Anew = numpy.delete(Anew, (k,), 2)
    Bnew = numpy.delete(Bnew, (l,), 1)
    Bnew = numpy.delete(Bnew, (l,), 2)

    dtype = (A[0, 0] * B[0, 0]).dtype
    C = numpy.zeros(tuple(A.shape[:-2]) + (nC, nC), dtype=dtype)
    C[..., :nA - 1, :nA - 1] = Anew
    C[..., nA - 1:, nA - 1:] = Bnew
    return C


def innerconnect_s(
        S: NDArray[numpy.complex128],
        k: int,
        l: int,
        ) -> NDArray[numpy.complex128]:
    """
    TODO
    freq x ...  x n x n

    Based on skrf implementation


    Connect two ports of a single n-port network's s-matrix.
    Specifically, connect port `k`  to port `l` on `S`. This results in
    a (n-2)-port network.

    Assumes same reference impedance for both `k` and `l`; may need to
    connect an "impedance mismatch" thru element first!

    Args:
        S: S-parameter matrix of `S`, shape is fxnxn
        k: port index on `S` (port indices start from 0)
        l: port index on `S`

    Returns:
        new S-parameter matrix

    Notes:
        - Compton, R.C., "Perspectives in microwave circuit analysis",
            doi:10.1109/MWSCAS.1989.101955
        - Filipsson, G., "A New General Computer Algorithm for S-Matrix Calculation
            of Interconnected Multiports",
            doi:10.1109/EUMA.1981.332972
    """
    if k > S.shape[-1] - 1 or l > S.shape[-1] - 1:
        raise ValueError("port indices are out of range")

    ll = slice(l, l + 1)
    kk = slice(k, k + 1)

    mkl = 1 - S[..., kk, ll]
    mlk = 1 - S[..., ll, kk]
    C = S + (
              S[..., kk, :] * S[..., :, l] * mlk
            + S[..., ll, :] * S[..., :, k] * mkl
            + S[..., kk, :] * S[..., l, l] * S[..., :, kk]
            + S[..., ll, :] * S[..., k, k] * S[..., :, ll]
        ) / (
            mlk * mkl - S[..., kk, kk] * S[..., ll, ll]
        )

    # remove connected ports
    C = numpy.delete(C, (k, l), 1)
    C = numpy.delete(C, (k, l), 2)

    return C


def s2abcd(
        S: NDArray[numpy.complex128],
        z0: NDArray[numpy.complex128],
        ) -> NDArray[numpy.complex128]:
    assert numpy.array_equal(S.shape[:2] == (2, 2))
    Z1, Z2 = z0
    cross = S[0, 1] * S[1, 0]

    T = numpy.empty_like(S, dtype=complex)
    T[0, 0, :] = (Z1.conj() + S[0, 0] * Z1) * (1 - S[1, 1]) + cross * Z1    # A numerator
    T[0, 1, :] = (Z1.conj() + S[0, 0] * Z1) * (Z1.conj() + S[1, 1] * Z2) - cross * Z1 * Z2  # B numerator
    T[1, 0, :] = (1 - S[0, 0]) * (1 - S[1, 1]) - cross                      # C numerator
    T[1, 1, :] = (1 - S[0, 0]) * (Z2.conj() + S[1, 1] * Z2) + cross * Z2    # D numerator
    det = 2 * S[1, 0] * numpy.sqrt(Z1.real * Z2.real)
    T /= det
    return T


def generalize_S(
        S: NDArray[numpy.complex128],
        r0: float,
        z0: NDArray[numpy.complex128],
        ) -> NDArray[numpy.complex128]:
    g = (z0 - r0) / (z0 + r0)
    D = numpy.diag((1 - g) / numpy.abs(1 - g.conj()) * numpy.sqrt(1 - numpy.abs(g * g.conj())))
    G = numpy.diag(g)
    U = numpy.eye(S.shape[-1]).reshape((S.ndim - 2) * (1,) + (S.shape[-2], S.shape[-1]))
    S_gen = pinv(D.conj()) @ (S - G.conj()) @ pinv(U - G @ S) @ D
    return S_gen


def change_R0(
        S: NDArray[numpy.complex128],
        r1: float,
        r2: float,
        ) -> NDArray[numpy.complex128]:
    g = (r2 - r1) / (r2 + r1)
    U = numpy.eye(S.shape[-1]).reshape((S.ndim - 2) * (1,) + (S.shape[-2], S.shape[-1]))
    G = U * g
    S_r2 = (S - G) @ pinv(U - G @ S)
    return S_r2

# Zc = numpy.sqrt(B / C)
# gamma = numpy.arccosh(A) / L_TL
# n_eff = -1j * gamma * c_light / (2 * pi * f)
# n_eff_grp = n_eff + f * diff(n_eff) / diff(f)
# attenuation = (1 - S[0, 0] * S[0, 0].conj()) / (S[1, 0] * S[1, 0].conj())
# R = numpy.real(gamma * Zc)
# C = numpy.real(gamma / Zc)
# L = numpy.imag(gamma * Zc) / (-1j * 2 * pi * f)
# G = numpy.imag(gamma / Zc) / (-1j * 2 * pi * f)
