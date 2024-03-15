# Based on scripts from Andy H. va rfcafe

# IEEE TRANSACTIONS ON MICROWAVE THEORY AND TECHNIQUES. VOL 42, NO 2. FEBRUARY 1994
# Conversions Between S, Z, Y, h, ABCD, and T Parameters which are Valid for Complex Source and Load Impedances
# Dean A. Frickey, Member, EEE
# Tables I and II

import numpy


def s_to_z(s, z0):
    """
    Scattering (S) to Impedance (Z)

    Args:
        s: The scattering matrix.
        z0: The port impedances (Ohms).

    Returns:
        The impedance matrix.
    """
    z0c = numpy.conj(z0)

    z = numpy.empty([2, 2], dtype=complex)
    z[0, 0] = (z0c[0] + s[0, 0] * z0[0]) * (1 - s[1, 1]) + s[0, 1] * s[1, 0] * z0[0]
    z[0, 1] = 2 * s[0, 1] * numpy.sqrt(z0[0].real * z0[1].real)
    z[1, 0] = 2 * s[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    z[1, 1] = (1 - s[0, 0]) * (z0c[1] + s[1, 1] * z0[1]) + s[0, 1] * s[1, 0] * z0[1]

    z /= (1 - s[0, 0]) * (1 - s[1, 1]) - s[0, 1] * s[1, 0]
    return z


def z_to_s(z, z0):
    """
    Impedance (Z) to Scattering (S)

    Args:
        z: The impedance matrix.
        z0: The port impedances (Ohms).

    Returns:
        The scattering matrix.
    """
    z0c = numpy.conj(z0)

    s = numpy.empty([2, 2], dtype=complex)
    s[0, 0] = (z[0, 0] - z0c[0]) * (z[1, 1] + z0[1]) - z[0, 1] * z[1, 0]
    s[0, 1] = 2 * z[0, 1] * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 0] = 2 * z[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 1] = (z[0, 0] + z0[0]) * (z[1, 1] - z0c[1]) - z[0, 1] * z[1, 0]

    s /= (z[0, 0] + z0[0]) * (z[1, 1] + z0[1]) - z[0, 1] * z[1, 0]
    return s


def s_to_y(s, z0):
    """
    Scattering (S) to Admittance (Y)

    Args:
        s: The scattering matrix.
        z0: The port impedances (Ohms).

    Returns:
        The admittance matrix.
    """
    z0c = numpy.conj(z0)

    y = numpy.empty([2, 2], dtype=complex)
    y[0, 0] = (1 - s[0, 0]) * (z0c[1] + s[1, 1] * z0[1]) + s[0,1] * s[1, 0] * z0[1]
    y[0, 1] = -2 * s[0,1] * numpy.sqrt(z0[0].real * z0[1].real)
    y[1, 0] = -2 * s[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    y[1, 1] = (z0c[0] + s[0, 0] * z0[0]) * (1 - s[1,1]) + s[0,1] * s[1, 0] * z0[0]

    y /= (z0c[0] + s[0, 0] * z0[0]) * (z0c[1] + s[1, 1] * z0[1]) - s[0,1] * s[1, 0] * z0[0] * z0[1]
    return y


def y_to_s(y, z0):
    """
    Admittance (Y) to Scattering (S)

    Args:
        y: The admittance matrix.
        z0: The port impedances (Ohms).

    Returns:
        The scattering matrix.
    """
    z0c = numpy.conj(z0)

    s = numpy.empty([2, 2], dtype=complex)
    s[0, 0] = (1 - y[0, 0] * z0c[0]) * (1 + y[1, 1] * z0[1]) + y[0,1] * y[1, 0] * z0c[0] * z0[1]
    s[0, 1] = -2 * y[0,1] * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 0] = -2 * y[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 1] = (1 + y[0, 0] * z0[0]) * (1 - y[1,1] * z0c[1]) + y[0,1] * y[1, 0] * z0[0] * z0c[1]

    s /= (1 + y[0, 0] * z0[0]) * (1 + y[1, 1] * z0[1]) - y[0,1] * y[1, 0] * z0[0] * z0[1]
    return s


def s_to_h(s, z0):
    """
    Scattering (S) to Hybrid (H)

    Args:
        s: The scattering matrix.
        z0: The port impedances (Ohms).

    Returns:
        The hybrid matrix.
    """
    z0c = numpy.conj(z0)

    h = numpy.empty([2, 2], dtype=complex)
    h[0, 0] = (z0c[0] + s[0, 0] * z0[0]) * (z0c[1] + s[1, 1] * z0[1]) - s[0,1] * s[1, 0] * z0[0] * z0[1]
    h[0, 1] = 2 * s[0,1] * numpy.sqrt(z0[0].real * z0[1].real)
    h[1, 0] = -2 * s[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    h[1, 1] = (1 - s[0, 0]) * (1 - s[1,1]) - s[0,1] * s[1, 0]

    h /= (1 - s[0, 0]) * (z0c[1] + s[1, 1] * z0[1]) + s[0,1] * s[1, 0] * z0[1]
    return h


def h_to_s(h, z0):
    """
    Hybrid (H) to Scattering (S)

    Args:
        h: The hybrid matrix.
        z0: The port impedances (Ohms).

    Returns:
        The scattering matrix.
    """
    z0c = numpy.conj(z0)

    s = numpy.empty([2, 2], dtype=complex)
    s[0, 0] = (h[0, 0] - z0c[0]) * (1 + h[1, 1] * z0[1]) - h[0,1] * h[1, 0] * z0[1]
    s[0, 1] = 2 * h[0,1] * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 0] = -2 * h[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 1] = (z0[0] + h[0, 0]) * (1 - h[1,1] * z0c[1]) + h[0,1] * h[1, 0] * z0c[1]

    s /= (z0[0] + h[0, 0]) * (1 + h[1, 1] * z0[1]) - h[0,1] * h[1, 0] * z0[1]
    return s


def s_to_abcd(s, z0):
    """
    Scattering to Chain (ABCD)

    Args:
        s: The scattering matrix.
        z0: The port impedances (Ohms).

    Returns:
        The chain matrix.
    """
    z0c = numpy.conj(z0)

    ans = numpy.empty([2, 2], dtype=complex)
    ans[0, 0] = (z0c[0] + s[0, 0] * z0[0]) * (1 - s[1, 1]) + s[0,1] * s[1, 0] * z0[0]
    ans[0, 1] = (z0c[0] + s[0, 0] * z0[0]) * (z0c[1] + s[1,1] * z0[1]) - s[0,1] * s[1, 0] * z0[0] * z0[1]
    ans[1, 0] = (1 - s[0, 0]) * (1 - s[1, 1]) - s[0,1] * s[1, 0]
    ans[1, 1] = (1 - s[0, 0]) * (z0c[1] + s[1,1] * z0[1]) + s[0,1] * s[1, 0] * z0[1]

    ans /= 2 * s[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    return ans


def abcd_to_s(abcd, z0):
    """
    Chain (ABCD) to Scattering (S)

    Args:
        abcd: The chain matrix.
        z0: The port impedances (Ohms).

    Return:
        The scattering matrix.
    """
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]

    z0c = numpy.conj(z0)

    s = numpy.empty([2, 2], dtype=complex)
    s[0, 0] = A * z0[1] + B - C * z0c[0] * z0[1] - D * z0c[0]
    s[0, 1] = 2 * (A * D - B * C) * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 0] = 2 * numpy.sqrt(z0[0].real * z0[1].real)
    s[1, 1] = -A * z0c[1] + B - C * z0[0] * z0c[1] + D * z0[0]

    s /= A * z0[1] + B + C * z0[0] * z0[1] + D * z0[0]
    return s


def t_to_z(t, z0):
    """
    Chain Transfer (T) to Impedance (Z)

    Args:
        t: The chain transfer matrix.
        z0: The port impedances (Ohms).

    Returns:
        The impedance matrix.
    """
    z0c = numpy.conj(z0)

    z = numpy.empty([2, 2], dtype=complex)
    z[0, 0] = z0c[0] * (t[0, 0] + t[0, 1]) + z0[0] * (t[1, 0] + t[1,1])
    z[0, 1] = 2 * numpy.sqrt(z0[0].real * z0[1].real) * (t[0, 0] * t[1,1] - t[0,1] * t[1, 0])
    z[1, 0] = 2 * numpy.sqrt(z0[0].real * z0[1].real)
    z[1, 1] = z0c[1] * (t[0, 0] - t[1, 0]) - z0[1] * (t[0,1] - t[1,1])

    z /= t[0, 0] + t[0, 1] - t[1, 0] - t[1,1]
    return z


def z_to_t(z, z0):
    """
    Impedance (Z) to Chain Transfer (T)

    Args:
        z: The impedance matrix.
        z0: The port impedances (Ohms).

    Returns:
        The chain transfer matrix.
    """
    z0c = numpy.conj(z0)

    t = numpy.empty([2, 2], dtype=complex)
    t[0, 0] = (z[0, 0] + z0[0]) * (z[1, 1] + z0[1]) - z[0,1] * z[1, 0]
    t[0, 1] = (z[0, 0] + z0[0]) * (z0c[1] - z[1,1]) + z[0,1] * z[1, 0]
    t[1, 0] = (z[0, 0] - z0c[0]) * (z[1, 1] + z0[1]) - z[0,1] * z[1, 0]
    t[1, 1] = (z0c[0] - z[0, 0]) * (z[1,1] - z0c[1]) + z[0,1] * z[1, 0]

    t /= 2 * z[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    return t


def t_to_y(t, z0):
    """
    Chain Transfer (T) to Admittance (Y)

    Args:
        t: The chain transfer matrix.
        z0: The port impedances (Ohms).

    Returns:
        The admittance matrix.
    """
    z0c = numpy.conj(z0)

    y = numpy.empty([2, 2], dtype=complex)
    y[0, 0] = z0c[1] * (t[0, 0] - t[1, 0]) - z0[1] * (t[0, 1] - t[1,1])
    y[0, 1] = -2 * numpy.sqrt(z0[0].real * z0[1].real) * (t[0, 0] * t[1,1] - t[0,1] * t[1, 0])
    y[1, 0] = -2 * numpy.sqrt(z0[0].real * z0[1].real)
    y[1, 1] = z0c[0] * (t[0, 0] + t[0,1]) + z0[0] * (t[1, 0] + t[1,1])

    y /= t[0, 0] * z0c[0] * z0c[1] - t[0, 1] * z0c[0] * z0[1] + t[1, 0] * z0[0] * z0c[1] - t[1,1] * z0[0] * z0[1]
    return y


def y_to_t(y, z0):
    """
    Admittance (Y) to Chain Transfer (T)

    Args:
        y: The admittance matrix.
        z0: The port impedances (Ohms).

    Returns:
        The chain transfer matrix.
    """
    z0c = numpy.conj(z0)

    t = numpy.empty([2, 2], dtype=complex)
    t[0, 0] = (-1 - y[0, 0] * z0[0]) * (1 + y[1, 1] * z0[1]) + y[0,1] * y[1, 0] * z0[0] * z0[1]
    t[0, 1] = (1 + y[0, 0] * z0[0]) * (1 - y[1,1] * z0c[1]) + y[0,1] * y[1, 0] * z0[0] * z0c[1]
    t[1, 0] = (y[0, 0] * z0c[0] - 1) * (1 + y[1, 1] * z0[1]) - y[0,1] * y[1, 0] * z0c[0] * z0[1]
    t[1, 1] = (1 - y[0, 0] * z0c[0]) * (1 - y[1,1] * z0c[1]) - y[0,1] * y[1, 0] * z0c[0] * z0c[1]

    t /= 2 * y[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    return t


def t_to_h(t, z0):
    """
    Chain Transfer (T) to Hybrid (H)

    Args:
        t: The chain transfer matrix.
        z0: The port impedances (Ohms).

    Returns:
        The hybrid matrix.
    """
    z0c = numpy.conj(z0)


    h = numpy.empty([2, 2], dtype=complex)
    h[0, 0] = z0c[1]*(t[0, 0] * z0c[0] + t[1, 0] * z0[0]) - z0[1] * (t[0, 1] * z0c[0] + t[1,1] * z0[0])
    h[0, 1] = 2 * numpy.sqrt(z0[0].real * z0[1].real) * (t[0, 0] * t[1,1] - t[0,1] * t[1, 0])
    h[1, 0] = -2 * numpy.sqrt(z0[0].real * z0[1].real)
    h[1, 1] = t[0, 0] + t[0,1] - t[1, 0] - t[1,1]

    h /= z0c[1] * (t[0, 0] - t[1, 0]) - z0[1] * (t[0, 1] - t[1,1])
    return h


def h_to_t(h, z0):
    """
    Hybrid (H) to Chain Transfer (T)

    Args:
        t: The hybrid matrix.
        z0: The port impedances (Ohms).

    Returns:
        The chain transfer matrix.
    """
    z0c = numpy.conj(z0)

    t = numpy.empty([2, 2], dtype=complex)
    t[0, 0] = (-h[0, 0] - z0[0]) * (1 + h[1, 1] * z0[1]) + h[0,1] * h[1, 0] * z0[1]
    t[0, 1] = (h[0, 0] + z0[0]) * (1 - h[1,1] * z0c[1]) + h[0,1] * h[1, 0] * z0c[1]
    t[1, 0] = (z0c[0] - h[0, 0]) * (1 + h[1, 1] * z0[1]) + h[0,1] * h[1, 0] * z0[1]
    t[1, 1] = (h[0, 0] - z0c[0]) * (1 - h[1,1] * z0c[1]) + h[0,1] * h[1, 0] * z0c[1]

    t /= 2 * h[1, 0] * numpy.sqrt(z0[0].real * z0[1].real)
    return t


def t_to_abcd(t, z0):
    """
    Chain Transfer (T) to Chain (ABCD)

    Args:
        t: The chain transfer matrix.
        z0: The port impedances (Ohms).

    Returns:
        The chain matrix.
    """
    z0c = numpy.conj(z0)
    ans = numpy.empty([2, 2], dtype=complex)
    ans[0, 0] = z0c[0] * (t[0, 0] + t[0, 1]) + z0[0] * (t[1, 0] + t[1, 1])
    ans[0, 1] = z0c[1] * (t[0, 0] * z0c[0] + t[1, 0] * z0[0]) - z0[1] * (t[0, 1] * z0c[0] + t[1, 1] * z0[0])
    ans[1, 0] = t[0, 0] + t[0, 1] - t[1, 0] - t[1, 1]
    ans[1, 1] = z0c[1] * (t[0, 0] - t[1, 0]) - z0[1] * (t[0, 1] - t[1, 1])

    ans /= 2 * numpy.sqrt(z0[0].real * z0[1].real)
    return ans


def abcd_to_t(abcd, z0):
    """
    Chain (ABCD) to Chain Transfer (T)

    Args:
        abcd: The chain matrix.
        z0: The port impedances (Ohms).

    Returns:
        The chain transfer matrix.
    """
    # Break out the components
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]

    z0c = numpy.conj(z0)

    t = numpy.empty([2, 2], dtype=complex)
    t[0, 0] = A * z0[1] + B + C * z0[0] * z0[1] + D * z0[0]
    t[0, 1] = A * z0c[1] - B + C * z0[0] * z0c[1] - D * z0[0]
    t[1, 0] = A * z0[1] + B - C * z0c[0] * z0[1] - D * z0c[0]
    t[1, 1] = A * z0c[1] - B - C * z0c[0] * z0c[1] + D * z0c[0]

    t /= 2 * numpy.sqrt(z0[0].real * z0[1].real)
    return t
