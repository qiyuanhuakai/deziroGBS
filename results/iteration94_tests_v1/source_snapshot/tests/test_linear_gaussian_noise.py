import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.linear_gaussian_noise import normal_moments,integrated_linear_law,gaussian_linear_target,product_coupling_bound
from gbslab.rank_one_noise import integrated_finite_law
from gbslab.growing_normal_grid import growing_grid

def vacuum():return [np.array([1,0],complex).reshape(1,2,1)]*2

class LinearNoiseTests(unittest.TestCase):
    def test_wick_fourth_mixed(self):
        m=normal_moments([[F(2),F(1,3)],[F(1,3),F(3)]]);self.assertEqual(m((2,2)),6+F(2,9));self.assertEqual(m((3,1)),2);self.assertEqual(m((4,0)),12)
    def test_wick_odd_and_zero(self):
        m=normal_moments([[F(2),F(1,3)],[F(1,3),F(3)]]);self.assertEqual(m((0,0)),1);self.assertEqual(m((3,2)),0)
    def test_invalid_moment_index(self):
        with self.assertRaises(ValueError):normal_moments([[1]])((-1,))
    def test_two_independent_vacuum_modes(self):
        v=[[(1,0),(0,0)],[(0,0),(1,0)]];r=integrated_linear_law(vacuum(),v,[F(1,4),F(1,8)]);a,b=r['intervals'][0];q=F(1)/((1+F(1,8))*(1+F(1,32)));self.assertLessEqual(a*a,q);self.assertGreaterEqual(b*b,q)
    def test_redundant_drives_one_photon(self):
        t=[np.array([0,1],complex).reshape(1,2,1),vacuum()[0]];v=[[(1,0),(0,0)],[(1,0),(0,0)]];r=integrated_linear_law(t,v,[F(1,4),F(1,8)]);a,b=r['vacuum_intervals'][1];s=F(5,64);q=s*s/(1+2*s)**3;self.assertLessEqual(a*a,q);self.assertGreaterEqual(b*b,q)
    def test_rank_one_compatibility(self):
        t=[np.array([1,0,F(1,2)],complex).reshape(1,3,1)];old=integrated_finite_law(t,[(1,F(1,2))],F(1,4));new=integrated_linear_law(t,[[(1,F(1,2))]],[F(1,4)])
        for a,b in zip(old['intervals'],new['intervals']):self.assertLessEqual(max(a[0],b[0]),min(a[1],b[1]))
    def test_zero_scales(self):
        v=[[(1,0),(1,0)],[(F(1,2),0),(0,1)]];r=integrated_linear_law(vacuum(),v,[0,0]);self.assertEqual(r['intervals'],[(F(1),F(1)),(F(0),F(0)),(F(0),F(0)),(F(0),F(0))])
    def test_noise_covariance_cross_terms(self):
        v=[[(1,0),(1,0)],[(F(1,2),0),(0,1)]];r=gaussian_linear_target([[0,0],[0,0]],[[0,0],[0,0]],v,[F(1,4),F(1,8)]);self.assertEqual(r['covariance'][0][1],F(1,4));self.assertEqual(r['covariance'][0][3],F(1,32));self.assertEqual(r['covariance'][0][0],1+F(17,64))
    def test_product_coupling(self):
        v=[[(1,0),(0,0)],[(0,0),(1,0)]];r=gaussian_linear_target([[0,0],[0,0]],[[0,0],[0,0]],v,[F(1,4),F(1,8)]);g=[growing_grid(s,4,radius=2) for s in r['sigmas']];b=product_coupling_bound(r,g);self.assertEqual(b['upper'],sum(b['terms']));self.assertEqual(len(b['terms']),2)
    def test_invalid_drive_count(self):
        with self.assertRaises(ValueError):integrated_linear_law(vacuum(),[],[])
    def test_invalid_scale(self):
        with self.assertRaises(ValueError):integrated_linear_law(vacuum(),[[(1,0),(0,0)]],[1])
    def test_zero_input(self):
        with self.assertRaises(ValueError):integrated_linear_law([np.zeros((1,2,1),complex)],[[(1,0)]],[F(1,4)])
