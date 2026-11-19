import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.multidrive_gaussian_noise import integrated_linear_law,gaussian_linear_target,normal_moments
from gbslab.linear_gaussian_noise import integrated_linear_law as old_integral

DIR=[[(1,0),(0,0)],[(0,1),(0,0)],[(0,0),(1,0)],[(0,0),(0,1)]]
S=[F(1,8),F(3,16),F(1,4),F(1,8)]
def vacuum():return [np.array([1,0],complex).reshape(1,2,1)]*2
class MultidriveNoiseTests(unittest.TestCase):
    def overlap(self,a,b):
        for x,y in zip(a,b):self.assertLessEqual(max(x[0],y[0]),min(x[1],y[1]))
    def test_four_quadrature_vacuum_exact(self):
        r=integrated_linear_law(vacuum(),DIR,S);a,b=r['vacuum_intervals'][3];q=F(1)
        for s in S:q/=1+2*s*s
        self.assertLessEqual(a*a,q);self.assertGreaterEqual(b*b,q)
    def test_vacuum_matches_full_gaussian(self):
        a=integrated_linear_law(vacuum(),DIR,S);b=gaussian_linear_target([[0,0],[0,0]],[[0,0],[0,0]],DIR,S);self.overlap(a['intervals'],b['intervals'])
    def test_photon_with_two_quadratures(self):
        t=[np.array([0,1],complex).reshape(1,2,1),vacuum()[0]];r=integrated_linear_law(t,DIR,S);a,b=r['vacuum_intervals'][1];x,y=S[:2];q=(x*x/(1+2*x*x)+y*y/(1+2*y*y))**2/((1+2*x*x)*(1+2*y*y));self.assertLessEqual(a*a,q);self.assertGreaterEqual(b*b,q)
    def test_four_redundant_drives_reduce_rank_one(self):
        t=[np.array([1,1j],complex).reshape(1,2,1)];v=[[(1,0)]];a=integrated_linear_law(t,v*4,[F(1,8)]*4);b=old_integral(t,v,[F(1,4)]);self.overlap(a['intervals'],b['intervals'])
    def test_three_redundant_drives_reduce_rank_one(self):
        t=[np.array([0,1],complex).reshape(1,2,1)];v=[[(1,0)]];a=integrated_linear_law(t,v*3,[F(1,8),F(1,4),F(1,4)]);b=old_integral(t,v,[F(3,8)]);self.overlap(a['intervals'],b['intervals'])
    def test_zero_axes_reduce_old_two_drives(self):
        t=vacuum();a=integrated_linear_law(t,DIR,[S[0],0,S[2],0]);b=old_integral(t,[DIR[0],DIR[2]],[S[0],S[2]]);self.overlap(a['intervals'],b['intervals'])
    def test_drive_permutation(self):
        t=[np.array([1,1j],complex).reshape(1,2,1),vacuum()[0]];p=[3,1,0,2];a=integrated_linear_law(t,DIR,S);b=integrated_linear_law(t,[DIR[i] for i in p],[S[i] for i in p]);self.overlap(a['intervals'],b['intervals'])
    def test_scale_direction_equivalence(self):
        v=[[(2*a,2*b) for a,b in d] for d in DIR];a=integrated_linear_law(vacuum(),DIR,S);b=integrated_linear_law(vacuum(),v,[s/2 for s in S]);self.assertEqual(a['intervals'],b['intervals'])
    def test_four_distinct_mixed_moment(self):
        C=[[F(2) if i==j else F(i+j+1,20) for j in range(4)] for i in range(4)];m=normal_moments(C);self.assertEqual(m((1,1,1,1)),C[0][1]*C[2][3]+C[0][2]*C[1][3]+C[0][3]*C[1][2])
    def test_degenerate_sixth_moment(self):self.assertEqual(normal_moments([[F(1)]*4 for _ in range(4)])((2,2,1,1)),15)
    def test_invalid_drive_count(self):
        with self.assertRaises(ValueError):integrated_linear_law(vacuum(),DIR+[DIR[0]],S+[S[0]])
    def test_invalid_scale(self):
        with self.assertRaises(ValueError):integrated_linear_law(vacuum(),DIR,[-S[0],*S[1:]])
    def test_zero_input(self):
        with self.assertRaises(ValueError):integrated_linear_law([np.zeros((1,2,1),complex)],[[(1,0)]]*4,S)
