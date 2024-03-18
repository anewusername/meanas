import scipy
import numpy
from numpy.typing import ArrayLike, NDArray


#from simphony.elements import Model
#from simphony.netlist import Subcircuit
#from simphony.simulation import SweepSimulation
#
#from matplotlib import pyplot as plt
#
#
#class PeriodicLayer(Model):
#    def __init__(self, left_modes, right_modes, s_params):
#        self.left_modes = left_modes
#        self.right_modes = right_modes
#        self.left_ports = len(self.left_modes)
#        self.right_ports = len(self.right_modes)
#        self.normalize_fields()
#        self.s_params = s_params
#
#    def normalize_fields(self):
#        for mode in range(len(self.left_modes)):
#            self.left_modes[mode].normalize()
#        for mode in range(len(self.right_modes)):
#            self.right_modes[mode].normalize()
#
#
#class PeriodicEME:
#    def __init__(self, layers=[], num_periods=1):
#        self.layers = layers
#        self.num_periods = num_periods
#        self.wavelength = wavelength
#
#    def propagate(self):
#        wl = self.wavelength
#        if not len(self.layers):
#            raise Exception("Must place layers before propagating")
#
#        num_modes = max([l.num_modes for l in self.layers])
#        iface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode
#
#        eme = EME(layers=self.layers)
#        left, right = eme.propagate()
#        self.single_period = eme.s_matrix
#
#        period_layer = PeriodicLayer(left.modes, right.modes, self.single_period)
#        current_layer = PeriodicLayer(left.modes, right.modes, self.single_period)
#        interface = iface(right, left)
#
#        for _ in range(self.num_periods - 1):
#            current_layer.s_params = cascade(current_layer, interface, wl)
#            current_layer.s_params = cascade(current_layer, period_layer, wl)
#
#        self.s_params = current_layer.s_params
#
#
#class EME:
#    def __init__(self, layers=[]):
#        self.layers = layers
#        self.wavelength = None
#
#    def propagate(self):
#        layers = self.layers
#        wl = layers[0].wavelength if self.wavelength is None else self.wavelength
#        if not len(layers):
#            raise Exception("Must place layers before propagating")
#
#        num_modes = max([l.num_modes for l in layers])
#        iface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode
#
#        first_layer = layers[0]
#        current = Current(wl, first_layer)
#        interface = iface(first_layer, layers[1])
#
#        current.s = cascade(current, interface, wl)
#        current.right_pins = interface.right_pins
#
#        for index in range(1, len(layers) - 1):
#            layer1 = layers[index]
#            layer2 = layers[index + 1]
#            interface = iface(layer1, layer2)
#
#            current.s = cascade(current, layer1, wl)
#            current.right_pins = layer1.right_pins
#
#            current.s = cascade(current, interface, wl)
#            current.right_pins = interface.right_pins
#
#        last_layer = layers[-1]
#        current.s = cascade(current, last_layer, wl)
#        current.right_pins = last_layer.right_pins
#
#        self.s_matrix = current.s
#        return first_layer, last_layer
#
#
#def stack(sa, sb):
#    qab = numpy.eye() - sa.r11 @ sb.r11
#    qba = numpy.eye() - sa.r11 @ sb.r11
#    #s.t12 = sa.t12 @ numpy.pinv(qab) @ sb.t12
#    #s.r21 = sa.t12 @ numpy.pinv(qab) @ sb.r22 @ sa.t21 + sa.r22
#    #s.r12 = sb.t21 @ numpy.pinv(qba) @ sa.r11 @ sb.t12 + sb.r11
#    #s.t21 = sb.t21 @ numpy.pinv(qba) @ sa.t21
#    s.t12 = sa.t12 @ numpy.linalg.solve(qab, sb.t12)
#    s.r21 = sa.t12 @ numpy.linalg.solve(qab, sb.r22 @ sa.t21) + sa.r22
#    s.r12 = sb.t21 @ numpy.linalg.solve(qba, sa.r11 @ sb.t12) + sb.r11
#    s.t21 = sb.t21 @ numpy.linalg.solve(qba, sa.t21)
#    return s
#
#
#def cascade(first, second, wavelength):
#    circuit = Subcircuit("Device")
#
#    circuit.add([(first, "first"), (second, "second")])
#    for port in range(first.right_ports):
#        circuit.connect("first", "right" + str(port), "second", "left" + str(port))
#
#    simulation = SweepSimulation(circuit, wavelength, wavelength, num=1)
#    result = simulation.simulate()
#    return result.s
#
#
#class InterfaceSingleMode(Model):
#    def __init__(self, layer1, layer2, num_modes=1):
#        self.num_modes = num_modes
#        self.num_ports = 2 * num_modes
#        self.s = self.solve(layer1, layer2, num_modes)
#
#    def solve(self, layer1, layer2, num_modes):
#        nm = num_modes
#        s = numpy.zeros((2 * nm, 2 * nm), dtype=complex)
#
#        for ii, left_mode in enumerate(layer1.modes):
#            for oo, right_mode in enumerate(layer2.modes):
#                r, t = get_rt(left_mode, right_mode)
#                s[     oo, ii] = r
#                s[nm + oo, ii] = t
#
#        for ii, right_mode in enumerate(layer2.modes):
#            for oo, left_mode in enumerate(layer1.modes):
#                r, t = get_rt(right_mode, left_mode)
#                s[     oo, nm + ii] = t
#                s[nm + oo, nm + ii] = r
#        return s
#
#
#class InterfaceMultiMode(Model):
#    def __init__(self, layer1, layer2):
#        self.s = self.solve(layer1, layer2)
#
#    def solve(self, layer1, layer2):
#        n1p = layer1.num_modes
#        n2p = layer2.num_modes
#        num_ports = n1p + n2p
#        s = numpy.zeros((num_ports, num_ports), dtype=complex)
#
#        for l1p in range(n1p):
#            ts = get_t(l1p, layer1, layer2)
#            rs = get_r(l1p, layer1, layer2, ts)
#            s[n1p:, l1p] = ts
#            s[:n1p, l1p] = rs
#
#        for l2p in range(n2p):
#            ts = get_t(l2p, layer2, layer1)
#            rs = get_r(l2p, layer2, layer1, ts)
#            s[:n1p, n1p + l2p] = ts
#            s[n1p:, n1p + l2p] = rs
#
#        return s


def get_t(p, left, right):
    A = numpy.empty(left.num_modes, right.num_modes, dtype=complex)
    for i in range(left.num_modes):
        for k in range(right.num_modes):
            # TODO optimize loop
            A[i, k] = inner_product(right[k], left[i]) + inner_product(left[i], right[k])

    b = numpy.zeros(left.num_modes)
    b[p] = 2 * inner_product(left[p], left[p])

    x = numpy.linalg.solve(A, b)
    # NOTE: `A` does not depend on `p`, so it might make sense to partially precompute
    # the solution (pinv(A), or LU decomposition?)
    # Actually solve() can take multiple vectors, so just pass it something with the full diagonal?

    xx = numpy.matmul(numpy.linalg.pinv(A), b)        #TODO verify
    assert numpy.allclose(xx, x)
    return x


def get_r(p, left, right, t):
    r = numpy.empty(left.num_modes, dtype=complex)
    for ii in range(left.num_modes):
        r[ii] = sum((inner_product(right[kk], left[ii]) - inner_product(left[ii], right[kk])) * t[kk]
                    for kk in range(right.num_modes)
                   ) / (2 * inner_product(left[ii], left[ii]))
    return r


def get_rt(left, right):
    a = 0.5 * (inner_product(left, right) + inner_product(right, left))
    b = 0.5 * (inner_product(left, right) - inner_product(right, left))
    t = (a ** 2 - b ** 2) / a
    r = 1 - t / (a + b)
    return -r, t


def inner_product(left_E, right_H, dxes):
    #  ExHy' - EyHx'
    cross_z = left_E[0] * right_H[1].conj() - left_E[1] * right_H[0].conj()
#    cross_z = numpy.cross(left_E, numpy.conj(right_H), axisa=0, axisb=0, axisc=0)[2]
    return numpy.trapz(numpy.trapz(cross_z, dxes[0][0]), dxes[0][1]) / 2       # TODO might need cumsum on dxes


def propagation_matrix(mode_neffs: ArrayLike, wavelength: float, distance: float):
    eigenv = numpy.array(mode_neffs, copy=False) * 2 * numpy.pi / wavelength
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
    Anew = npy.delete(Anew, (k,), 1)
    Anew = npy.delete(Anew, (k,), 2)
    Bnew = npy.delete(Bnew, (l,), 1)
    Bnew = npy.delete(Bnew, (l,), 2)

    dtype = (A[0, 0] * B[0, 0]).dtype
    C = numpy.zeros(tuple(A.shape[:-2]) + (nn, nn), dtype=dtype)
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
    C = npy.delete(C, (k, l), 1)
    C = npy.delete(C, (k, l), 2)

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
    U = numpy.eye(S.shape[0])
    S_gen = pinv(D.conj()) @ (S - G.conj()) @ pinv(U - G @ S) @ D
    return S_gen


def change_R0(
        S: NDArray[numpy.complex128],
        r1: float,
        r2: float,
        ) -> NDArray[numpy.complex128]:
    g = (r2 - r1) / (r2 + r1)
    U = numpy.eye(S.shape[0])
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
