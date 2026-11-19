import math
import unittest
import numpy as np
from gbslab.gaussian import haar_unitary
from gbslab.meixner_probe import uniform_parameters, signed_degree_budget, threshold_projection


class MeixnerProbeTests(unittest.TestCase):
    def test_geometric_conditional_moments(self):
        b = .4
        x = np.arange(200, dtype=float)
        mu = (b/(1+b))**x/(1+b)
        for h, no, yes in [(x-b, -b, 1), (x*(x-1)/2-2*b*x+b*b, b*b, -b)]:
            self.assertAlmostEqual(float(mu@h), 0., delta=1e-15)
            self.assertAlmostEqual(h[0], no)
            self.assertAlmostEqual(float(mu[1:]@h[1:]/(1-mu[0])), yes)

    def test_degree_zero_product_reference(self):
        u = haar_unitary(4, np.random.default_rng(51))
        q = threshold_projection(u, .6, .35, 0)
        b = q['parameters']['b']
        self.assertAlmostEqual(q['raw'][0], (1+b)**-4)
        self.assertAlmostEqual(q['raw'][-1], (b/(1+b))**4)

    def test_degree_two_against_explicit_pnr_sum(self):
        # Independently enumerate the reference PNR grid and Wick coefficients.
        u = haar_unitary(2, np.random.default_rng(52))
        r, eta = .5, .4
        p = uniform_parameters(2,r,eta)
        b, a, n = p['b'], p['a'], p['nout']
        M2 = eta**2*math.sinh(r)**2*math.cosh(r)**2*np.abs(u@u.T)**2
        out = np.zeros(4)
        for x in range(100):
            for y in range(100):
                h1 = np.array([x-b, y-b])
                h2 = np.array([x*(x-1)/2-2*b*x+b*b,y*(y-1)/2-2*b*y+b*b])
                moment1 = n-b
                moment2 = (n-b)**2+.5*np.diag(M2)
                cross = (n-b)**2+M2[0,1]
                f = 1 + moment1*sum(h1)/p['v'] + moment2@h2/p['v']**2 + cross*np.prod(h1)/p['v']**2
                out[(x>0)+2*(y>0)] += b**(x+y)/(1+b)**(x+y+2)*f
        np.testing.assert_allclose(out, threshold_projection(u,r,eta,2)['raw'], atol=2e-14)

    def test_signed_normalization_repair_and_mode_permutation(self):
        u = haar_unitary(6, np.random.default_rng(53))
        for d in (0,1,2):
            q = threshold_projection(u,.9,.65,d)
            self.assertAlmostEqual(q['raw_mass'], 1.)
            self.assertAlmostEqual(q['distribution'].sum(), 1.)
            self.assertAlmostEqual(q['repair_l1'], 2*q['negative_mass'], delta=2e-14)
            perm = [3,0,5,2,1,4]
            p = threshold_projection(u[perm],.9,.65,d)
            indices = [sum(((mask>>j)&1)<<perm[j] for j in range(6)) for mask in range(64)]
            np.testing.assert_allclose(p['raw'],q['raw'][indices],atol=2e-14)

    def test_budget_meets_markov_bound_and_is_minimal(self):
        for m in (8,64,1024):
            p = signed_degree_budget(m,.6,.35)
            self.assertLessEqual(p['markov_bound'],.05)
            self.assertGreater(p['C']*p['q']**p['degree']/.1**2,.05)

    def test_invalid_parameter_guards(self):
        for eta in (0,1,-1):
            with self.assertRaises(ValueError):
                uniform_parameters(4,.6,eta)
        with self.assertRaises(ValueError):
            threshold_projection(np.eye(4),.6,.35,3)


if __name__ == '__main__':
    unittest.main()
