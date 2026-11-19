import unittest
import numpy as np
from gbslab.mps_overlap import compare_mps, normalized_right_mps, scaled_overlap


def dense(tensors):
    v = tensors[0][0]
    for a in tensors[1:]: v = np.tensordot(v, a, axes=(-1, 0))
    return v.ravel()


class MPSOverlapTests(unittest.TestCase):
    def random_mps(self, bonds, rng):
        return [rng.normal(size=(a, 3, b))+1j*rng.normal(size=(a, 3, b)) for a, b in zip(bonds[:-1], bonds[1:])]

    def test_dense(self):
        rng = np.random.default_rng(220903)
        for bonds in ([1, 1], [1, 2, 4, 1], [1, 3, 5, 4, 3, 1]):
            a, b = self.random_mps(bonds, rng), self.random_mps(bonds, rng)
            va, vb = dense(a), dense(b); f = abs(np.vdot(va, vb))**2/(np.vdot(va, va).real*np.vdot(vb, vb).real)
            r = compare_mps(a, b)
            self.assertAlmostEqual(r['fidelity_from_overlap'], f, places=12)
            self.assertAlmostEqual(r['pure_trace_distance_from_difference'], np.sqrt(1-f), places=12)
            self.assertAlmostEqual(r['log_norm_qr_residual_left'], 0., places=12)
            q, ln = normalized_right_mps(a)
            np.testing.assert_allclose(dense(q), va/np.linalg.norm(va), atol=1e-14)
            self.assertAlmostEqual(ln, np.log(np.linalg.norm(va)), places=12)

    def test_scale_phase_and_gauge(self):
        rng = np.random.default_rng(220904); a = self.random_mps([1, 3, 3, 1], rng)
        b = [x.copy() for x in a]
        gauge = np.eye(3) + .1*rng.normal(size=(3, 3))
        b[0] = b[0] @ gauge
        b[1] = np.tensordot(np.linalg.inv(gauge), b[1], axes=(1, 0))
        b = [x*1e150 for x in b]; b[0] *= np.exp(.7j)
        r = compare_mps(a, b)
        self.assertLess(r['pure_trace_distance_from_difference'], 1e-13)
        self.assertAlmostEqual(r['log_norm_squared_right']-r['log_norm_squared_left'], 6*np.log(1e150), places=10)

    def test_near_identity(self):
        a = [np.array([1., 0.]).reshape(1, 2, 1)]
        b = [np.array([1., 1e-10]).reshape(1, 2, 1)]
        r = compare_mps(a, b)
        self.assertEqual(r['fidelity_from_overlap'], 1.)
        self.assertAlmostEqual(r['pure_trace_distance_from_difference']/1e-10, 1., places=12)

    def test_orthogonal(self):
        a = [np.array([1., 0.]).reshape(1, 2, 1)]
        b = [np.array([0., 1.]).reshape(1, 2, 1)]
        r = compare_mps(a, b)
        self.assertEqual(r['fidelity_from_overlap'], 0.)
        self.assertAlmostEqual(r['pure_trace_distance_from_difference'], 1.)

    def test_zero_and_shapes(self):
        a = [np.ones((1, 2, 1))]
        with self.assertRaises(ValueError): compare_mps(a, [np.zeros((1, 2, 1))])
        with self.assertRaises(ValueError): scaled_overlap(a, [np.ones((1, 3, 1))])
        with self.assertRaises(ValueError): scaled_overlap([], a)
        with self.assertRaises(ValueError): scaled_overlap([np.ones((2, 2, 1))], a)


if __name__ == '__main__': unittest.main()
