from collections.abc import Sequence
import numpy
from numpy.typing import NDArray
from scipy import sparse

from ..fdmath import dx_lists2_t, vcfdfield2
from .waveguide_2d import inner_product


def get_tr(
        ehLs: Sequence[Sequence[vcfdfield2]],
        wavenumbers_L: Sequence[complex],
        ehRs: Sequence[Sequence[vcfdfield2]],
        wavenumbers_R: Sequence[complex],
        dxes: dx_lists2_t,
        ) -> tuple[NDArray[numpy.complex128], NDArray[numpy.complex128]]:
    nL = len(wavenumbers_L)
    nR = len(wavenumbers_R)
    A12 = numpy.zeros((nL, nR), dtype=complex)
    A21 = numpy.zeros((nL, nR), dtype=complex)
    B11 = numpy.zeros((nL,), dtype=complex)
    for ll in range(nL):
        eL, hL = ehLs[ll]
        B11[ll] = inner_product(eL, hL, dxes=dxes, conj_h=False)
        for rr in range(nR):
            eR, hR = ehRs[rr]
            A12[ll, rr] = inner_product(eL, hR, dxes=dxes, conj_h=False)    # TODO optimize loop?
            A21[ll, rr] = inner_product(eR, hL, dxes=dxes, conj_h=False)

    # tt0 = 2 * numpy.linalg.pinv(A21 + numpy.conj(A12))
    tt0, _resid, _rank, _sing = numpy.linalg.lstsq(A21 + A12, numpy.diag(2 * B11), rcond=None)

    U, st, V = numpy.linalg.svd(tt0)
    gain = st > 1
    st[gain] = 1 / st[gain]
    tt = U @ numpy.diag(st) @ V

    # rr = 0.5 * (A21 - numpy.conj(A12)) @ tt
    rr = numpy.diag(0.5 / B11) @ (A21 - A12) @ tt

    return tt, rr


def get_abcd(
        ehLs: Sequence[Sequence[vcfdfield2]],
        wavenumbers_L: Sequence[complex],
        ehRs: Sequence[Sequence[vcfdfield2]],
        wavenumbers_R: Sequence[complex],
        **kwargs,
        ) -> sparse.sparray:
    t12, r12 = get_tr(ehLs, wavenumbers_L, ehRs, wavenumbers_R, **kwargs)
    t21, r21 = get_tr(ehRs, wavenumbers_R, ehLs, wavenumbers_L, **kwargs)
    t21i = numpy.linalg.pinv(t21)
    A = t12 - r21 @ t21i @ r12
    B = r21 @ t21i
    C = -t21i @ r12
    D = t21i
    return sparse.block_array(((A, B), (C, D)))


def get_s(
        ehLs: Sequence[Sequence[vcfdfield2]],
        wavenumbers_L: Sequence[complex],
        ehRs: Sequence[Sequence[vcfdfield2]],
        wavenumbers_R: Sequence[complex],
        force_nogain: bool = False,
        force_reciprocal: bool = False,
        **kwargs,
        ) -> NDArray[numpy.complex128]:
    t12, r12 = get_tr(ehLs, wavenumbers_L, ehRs, wavenumbers_R, **kwargs)
    t21, r21 = get_tr(ehRs, wavenumbers_R, ehLs, wavenumbers_L, **kwargs)

    ss = numpy.block([[r12, t12],
                      [t21, r21]])

    if force_nogain:
        # force S @ S.H diagonal
        U, sing, V = numpy.linalg.svd(ss)
        ss = numpy.diag(sing) @ U @ V

    if force_reciprocal:
        ss = 0.5 * (ss + ss.T)

    return ss
