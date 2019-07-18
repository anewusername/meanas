import unittest
import numpy

from fdfd_tools import fdtd

class BasicTests():
    def test_initial_fields(self):
        # Make sure initial fields didn't change
        e0 = self.es[0]
        h0 = self.hs[0]
        mask = self.src_mask

        self.assertEqual(e0[mask], self.j_mag / self.epsilon[mask])
        self.assertFalse(e0[~mask].any())
        self.assertFalse(h0.any())


    def test_initial_energy(self):
        e0 = self.es[0]
        h0 = self.hs[0]
        h1 = self.hs[1]
        mask = self.src_mask[1]
        dxes = self.dxes if self.dxes is not None else tuple(tuple(numpy.ones(s) for s in e0.shape[1:]) for _ in range(2))
        dV = numpy.prod(numpy.meshgrid(*dxes[0], indexing='ij'), axis=0)
        u0 = self.j_mag * self.j_mag / self.epsilon[self.src_mask] * dV[mask]
        args = {'dxes': self.dxes,
                'epsilon': self.epsilon}

        # Make sure initial energy and E dot J are correct
        energy0 = fdtd.energy_estep(h0=h0, e1=e0, h2=self.hs[1], **args)
        e_dot_j_0 = fdtd.delta_energy_j(j0=(e0 - 0) * self.epsilon, e1=e0, dxes=self.dxes)
        self.assertEqual(energy0[mask], u0)
        self.assertFalse(energy0[~mask].any())
        self.assertEqual(e_dot_j_0[mask], u0)
        self.assertFalse(e_dot_j_0[~mask].any())


    def test_energy_conservation(self):
        e0 = self.es[0]
        u0 = fdtd.delta_energy_j(j0=(e0 - 0) * self.epsilon, e1=e0, dxes=self.dxes).sum()
        args = {'dxes': self.dxes,
                'epsilon': self.epsilon}

        for ii in range(1, 8):
            with self.subTest(i=ii):
                u_hstep = fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii], **args)
                u_estep = fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1], **args)
                self.assertTrue(numpy.allclose(u_hstep.sum(), u0))
                self.assertTrue(numpy.allclose(u_estep.sum(), u0))


    def test_poynting(self):
        args = {'dxes': self.dxes,
                'epsilon': self.epsilon}

        u_eprev = None
        for ii in range(1, 8):
            with self.subTest(i=ii):
                u_hstep = fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii], **args)
                u_estep = fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1], **args)

                du_half_h2e = u_estep - u_hstep
                div_s_h2e = self.dt * fdtd.poynting_divergence(e=self.es[ii], h=self.hs[ii], dxes=self.dxes)
                self.assertTrue(numpy.allclose(du_half_h2e, -div_s_h2e))

                if u_eprev is None:
                    u_eprev = u_estep
                    continue

                # previous half-step
                du_half_e2h = u_hstep - u_eprev
                div_s_e2h = self.dt * fdtd.poynting_divergence(e=self.es[ii-1], h=self.hs[ii], dxes=self.dxes)
                self.assertTrue(numpy.allclose(du_half_e2h, -div_s_e2h))
                u_eprev = u_estep


class Basic2DNoDXOnlyVacuum(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 1]
        dt = 0.5
        epsilon = numpy.ones(shape, dtype=float)
        j_mag = 32
        dxes = None

        src_mask = numpy.zeros_like(epsilon, dtype=bool)
        src_mask[1, 2, 2, 0] = True

        e = numpy.zeros_like(epsilon)
        h = numpy.zeros_like(epsilon)
        e[src_mask] = j_mag / epsilon[src_mask]
        es = [e]
        hs = [h]

        eh2h = fdtd.maxwell_h(dt=dt, dxes=dxes)
        eh2e = fdtd.maxwell_e(dt=dt, dxes=dxes)
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
        self.dxes = dxes
        self.src_mask = src_mask
        self.j_mag = j_mag


class Basic3DUniformDXOnlyVacuum(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 5]
        dt = 0.33
        epsilon = numpy.ones(shape, dtype=float)
        j_mag = 32
        dxes = tuple(tuple(numpy.ones(s) for s in shape[1:]) for _ in range(2))

        src_mask = numpy.zeros_like(epsilon, dtype=bool)
        src_mask[1, 2, 2, 0] = True

        e = numpy.zeros_like(epsilon)
        h = numpy.zeros_like(epsilon)
        e[src_mask] = j_mag / epsilon[src_mask]
        es = [e]
        hs = [h]

        eh2h = fdtd.maxwell_h(dt=dt, dxes=dxes)
        eh2e = fdtd.maxwell_e(dt=dt, dxes=dxes)
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
        self.dxes = dxes
        self.src_mask = src_mask
        self.j_mag = j_mag


class Basic3DUniformDX(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 5]
        dt = 0.33
        epsilon = numpy.full(shape, 2, dtype=float)
        j_mag = 32
        dxes = tuple(tuple(numpy.ones(s) for s in shape[1:]) for _ in range(2))

        src_mask = numpy.zeros_like(epsilon, dtype=bool)
        src_mask[1, 2, 2, 0] = True

        e = numpy.zeros_like(epsilon)
        h = numpy.zeros_like(epsilon)
        e[src_mask] = j_mag / epsilon[src_mask]
        es = [e]
        hs = [h]

        eh2h = fdtd.maxwell_h(dt=dt, dxes=dxes)
        eh2e = fdtd.maxwell_e(dt=dt, dxes=dxes)
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
        self.dxes = dxes
        self.src_mask = src_mask
        self.j_mag = j_mag

