import numpy
from numpy import sqrt, real, abs
from numpy.linalg import pinv


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

    I = numpy.eye(SL.shape[-1])[None, :, :]
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
    I = numpy.eye(SL.shape[-1])[None, :, :]
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
    nf = zref0.shape[0]
    s = numpy.empty((nf, 2, 2), dtype=complex)
    s[:, 0, 0] = zref1 - zref0
    s[:, 0, 1] = 2 * sqrt(zref0 * zref1)
    s[:, 1, 0] = s[:, 0, 1]
    s[:, 1, 1] = -s[:, 0, 0]

    s /= zref0 + zref1
    return s


