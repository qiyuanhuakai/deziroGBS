import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.normal_grid_certificate import arctan_interval,pi_interval,normal_cdf,normal_grid,displacement_grid_bound,sample_grid_threshold


class NormalGridTests(unittest.TestCase):
    def test_pi_rational_bounds(self):
        a,b=pi_interval();self.assertTrue(F(314159265358979323846,10**20)<a<b<F(314159265358979323847,10**20));self.assertLess(b-a,F(1,10**200))
    def test_cdf_symmetry_and_center(self):
        self.assertEqual(normal_cdf(0),(F(1,2),F(1,2)))
        for x in (F(1,8),F(1),F(4)):
            a,b=normal_cdf(x);self.assertEqual(normal_cdf(-x),(1-b,1-a));self.assertTrue(F(1,2)<a<=b<1)
    def test_cdf_known_value(self):
        a,b=normal_cdf(1);self.assertTrue(F(841344746068542948,10**18)<a<=b<F(841344746068542949,10**18))
    def test_weights_keep_tail_and_normalization(self):
        g=normal_grid(F(1,4),16);self.assertEqual(len(g['nodes']),17);self.assertEqual(g['nodes'][-1],0);self.assertGreater(g['tail_interval'][0],0);self.assertLessEqual(sum(a for a,b in g['weight_intervals']),1);self.assertGreaterEqual(sum(b for a,b in g['weight_intervals']),1)
    def test_grid_refinement_bound(self):
        a=displacement_grid_bound(F(1,2),normal_grid(F(1,2),16));b=displacement_grid_bound(F(1,2),normal_grid(F(1,2),64));self.assertLess(b,a)
    def test_selector_reproducibility(self):
        t=[np.array([1,0,0,0],complex).reshape(1,4,1)];g=normal_grid(F(1,4),16);self.assertEqual(sample_grid_threshold(t,g,12,np.random.default_rng(19)),sample_grid_threshold(t,g,12,np.random.default_rng(19)))
    def test_guards(self):
        for sigma,cells in ((0,16),(1,16),(F(1,4),3),(F(1,4),1024)):
            with self.assertRaises(ValueError):normal_grid(sigma,cells)
        with self.assertRaises(ValueError):normal_cdf(5)
        with self.assertRaises(ValueError):arctan_interval(F(1))
        with self.assertRaises(ValueError):displacement_grid_bound(1,normal_grid(F(1,4),16))

if __name__=='__main__':unittest.main()
