import unittest
import numpy

from fdfd_tools import fdtd

class TestBasic2D(unittest.TestCase):
    def setUp(self):
        shape = [3, 5, 5, 1]
        dt = 0.5
        epsilon = numpy.ones(shape, dtype=float)

        src_mask = numpy.zeros_like(epsilon, dtype=bool)
        src_mask[1, 2, 2, 0] = True

        e = numpy.zeros_like(epsilon)
        h = numpy.zeros_like(epsilon)
        e[src_mask] = 32
        es = [e]
        hs = [h]

        eh2h = fdtd.maxwell_h(dt=dt)
        eh2e = fdtd.maxwell_e(dt=dt)
        for _ in range(9):
            e = e.copy()
            h = h.copy()
            eh2h(e, h)
            eh2e(e, h, epsilon)
            es.append(e)
            hs.append(h)

        self.es = es
        self.hs = hs
        self.dt = dt
        self.epsilon = epsilon
        self.src_mask = src_mask

    def test_initial_fields(self):
        # Make sure initial fields didn't change
        e0 = self.es[0]
        h0 = self.hs[0]
        self.assertEqual(e0[1, 2, 2, 0], 32)

        self.assertFalse(e0[~self.src_mask].any())
        self.assertFalse(h0.any())


    def test_initial_energy(self):
        e0 = self.es[0]
        h0 = self.hs[0]
        h1 = self.hs[1]
        mask = self.src_mask[1]

        # Make sure initial energy and E dot J are correct
        energy0 = fdtd.energy_estep(h0=h0, e1=e0, h2=self.hs[1])
        e_dot_j_0 = fdtd.delta_energy_j(j0=e0 - 0, e1=e0)
        self.assertEqual(energy0[mask], 32 * 32)
        self.assertFalse(energy0[~mask].any())
        self.assertEqual(e_dot_j_0[mask], 32 * 32)
        self.assertFalse(e_dot_j_0[~mask].any())


    def test_energy_conservation(self):
        for ii in range(1, 8):
            with self.subTest(i=ii):
                u_estep = fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1])
                u_hstep = fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii])
                self.assertTrue(numpy.allclose(u_estep.sum(), 32 * 32))
                self.assertTrue(numpy.allclose(u_hstep.sum(), 32 * 32))
