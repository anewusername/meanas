# IEEE TRANSACTIONS ON MICROWAVE THEORY AND TECHNIQUES. VOL 42, NO 2. FEBRUARY 1994
# Conversions Between S, Z, Y, h, ABCD, and T Parameters which are Valid for Complex Source and Load Impedances
# Dean A. Frickey, Member, EEE

# Tables I and II

import numpy as np
from two_port_conversions import *

""" Testing """
# Analog Devices - HMC455LP3 - S parameters
# High output IP3 GaAs InGaP Heterojunction Bipolar Transistor

# MHz S (Magntidue and Angle (deg))
# 1487.273 0.409 160.117 4.367 163.864 0.063 115.967 0.254 -132.654

s11 = 0.409 * np.exp(1j * np.radians(160.117))
s12 = 4.367 * np.exp(1j * np.radians(163.864))
s21 = 0.063 * np.exp(1j * np.radians(115.967))
s22 = 0.254 * np.exp(1j * np.radians(-132.654))

s_orig = np.array([[s11, s12], [s21, s22]])

# Data specified at 50 Ohms (adding small complex component to test conversions)
z1, z2 = 50 + 0.01j, 50 - 0.02j
z0 = np.array([z1, z2])

""" Conversions """
print(f'Original S: \n{s_orig}\n')

# S --> Z --> T --> Z --> S
z = s_to_z(s_orig, z0)
t = z_to_t(z, z0)
z = t_to_z(t, z0)
s = z_to_s(z, z0)
print(f'Test (S --> Z --> T --> Z --> S): \n{s}\n')

# S --> Y --> T --> Y --> S
y = s_to_y(s_orig, z0)
t = y_to_t(y, z0)
y = t_to_y(t, z0)
s = y_to_s(y, z0)
print(f'Test (S --> Y --> T --> Y --> S): \n{s}\n')

# S --> H --> T --> H --> S
h = s_to_h(s_orig, z0)
t = h_to_t(h, z0)
h = t_to_h(t, z0)
s = h_to_s(h, z0)
print(f'Test (S --> H --> T --> H --> S): \n{s}\n')

# S --> ABCD --> T --> ABCD --> S
abcd = s_to_abcd(s_orig, z0)
t = abcd_to_t(abcd, z0)
abcd = t_to_abcd(t, z0)
s = abcd_to_s(abcd, z0)
print(f'Test (S --> ABCD --> T --> ABCD --> S): \n{s}\n')

