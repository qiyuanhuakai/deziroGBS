import unittest
from fractions import Fraction as F
from gbslab.clipped_normal_grid import boundary_grid,boundary_displacement_bound
from gbslab.normal_grid_certificate import normal_grid,displacement_grid_bound


class ClippedNormalGridTests(unittest.TestCase):
    def test_tail_mass_preserved(self):
        g=boundary_grid(F(1,2),16);old=normal_grid(F(1,2),16);self.assertEqual(g['nodes'][-2:],[-2,2]);self.assertEqual(g['weight_intervals'][-2:],[(old['tail_interval'][0]/2,old['tail_interval'][1]/2)]*2);self.assertEqual(len(g['nodes']),18)
    def test_positive_tail_distance_moment(self):
        g=boundary_grid(F(1,4),64);a,b=g['tail_mean_excess_interval'];self.assertTrue(F(357,10**8)<a<=b<F(358,10**8))
    def test_bound_improves_fixed_zero_tail_bound(self):
        for lam in (0,F(1,2),F(3,4)):
            for sigma in (F(1,4),F(1,2)):
                for cells in (16,64,256):self.assertLess(boundary_displacement_bound(lam,boundary_grid(sigma,cells)),displacement_grid_bound(lam,normal_grid(sigma,cells)))
    def test_radius_density_is_strictly_positive(self):
        a,b=boundary_grid(F(1,4),16)['standard_density_at_radius_interval'];self.assertTrue(F(1338,10**7)<a<=b<F(1339,10**7))
    def test_selector_is_normalized(self):
        g=boundary_grid(F(1,2),256);self.assertEqual(float.fromhex(g['cdf_hex'][-1]),1);self.assertTrue(all(float.fromhex(x)>=0 for x in g['probabilities_hex']))
    def test_invalid_pairing_rejected(self):
        with self.assertRaises(ValueError):boundary_displacement_bound(1,boundary_grid(F(1,4),16))

if __name__=='__main__':unittest.main()
