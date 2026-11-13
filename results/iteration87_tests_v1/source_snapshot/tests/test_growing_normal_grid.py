import unittest
from fractions import Fraction as F
from gbslab.growing_normal_grid import scaled_exp_minus,growing_cdf,scheduled_radius,growing_grid,growing_displacement_bound

class GrowingGridTests(unittest.TestCase):
    def test_schedule(self):self.assertEqual([scheduled_radius(n) for n in (16,64,256,1024)],[3,4,5,6])
    def test_invalid_schedule(self):
        for n in (0,4,32,4096,16.0):
            with self.assertRaises(ValueError):scheduled_radius(n)
    def test_cdf_symmetry_and_mass(self):
        for x in (F(0),F(4),F(5),F(6),F(8)):
            a,b=growing_cdf(x);c,d=growing_cdf(-x);self.assertEqual((c,d),(1-b,1-a));self.assertTrue(0<=a<=b<=1)
    def test_cdf_far_tail_nonzero(self):
        a,b=growing_cdf(F(-8));self.assertGreater(a,0);self.assertLess(b,F(1,10**14))
    def test_exponential_semigroup(self):
        for x in (F(5),F(9),F(32),F(64)):
            a,b=scaled_exp_minus(x);c,d=scaled_exp_minus(x/2);self.assertLessEqual(a,d*d);self.assertGreaterEqual(b,c*c);self.assertGreater(a,0)
    def test_invalid_domain(self):
        for fn,x in ((scaled_exp_minus,-1),(scaled_exp_minus,65),(growing_cdf,9)):
            with self.assertRaises(ValueError):fn(x)
    def test_mass_and_boundary(self):
        g=growing_grid(F(3,8),16);self.assertEqual(g['nodes'][-2:],[-F(9,8),F(9,8)]);self.assertEqual(len(g['nodes']),18);self.assertLessEqual(sum(a for a,b in g['weight_intervals']),1);self.assertGreaterEqual(sum(b for a,b in g['weight_intervals']),1)
    def test_bound_decreases(self):
        bounds=[growing_displacement_bound(F(5,8),growing_grid(F(3,8),n)) for n in (16,64,256)]
        self.assertTrue(all(a>b for a,b in zip(bounds,bounds[1:])))
    def test_radius_four_selector_regression(self):
        from gbslab.clipped_normal_grid import boundary_grid
        for sigma in (F(1,4),F(1,2)):
            a=growing_grid(sigma,16,radius=4);b=boundary_grid(sigma,16);self.assertEqual(a['probabilities_hex'],b['probabilities_hex']);self.assertEqual(a['cdf_hex'],b['cdf_hex'])
