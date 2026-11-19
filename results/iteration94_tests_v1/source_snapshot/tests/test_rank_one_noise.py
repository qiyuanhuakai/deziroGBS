import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.rank_one_noise import integrated_finite_law,gaussian_rank_one,coupling_bound
from gbslab.growing_normal_grid import growing_grid
from gbslab.real_noise_integral import integrated_vacuum

def pair(lam=F(1,2),d=4,phase=1):
    a=np.zeros((1,d,d),complex);b=np.zeros((d,d,1),complex)
    for k in range(d):a[0,k,k]=complex(float(lam))**k*phase**k;b[k,k,0]=1
    return [a,b]

class RankOneNoiseTests(unittest.TestCase):
    def enclosed(self,x,p):self.assertLessEqual(p[0],x);self.assertLessEqual(x,p[1])
    def test_vacuum_common_normal(self):
        r=integrated_finite_law(pair(F(0)),[(1,0),(1,0)],F(1,4))
        for mask in range(4):
            p=r['vacuum_intervals'][mask];q=F(1)/(1+F(1,8)*mask.bit_count());self.assertLessEqual(p[0]**2,q);self.assertGreaterEqual(p[1]**2,q)
    def test_entangled_zero_noise_closed_form(self):
        r=integrated_finite_law(pair(),[(1,0),(0,1)],0);p=F(3,4)/(1-F(1,4)**4);self.enclosed(p,r['intervals'][0]);self.enclosed(0,r['intervals'][1]);self.enclosed(0,r['intervals'][2]);self.enclosed(1-p,r['intervals'][3])
    def test_complex_vacuum_norm(self):
        r=integrated_finite_law(pair(F(0)),[(1,0),(F(1,2),F(1,2))],F(1,4));a,b=r['intervals'][0];q=F(1)/(1+F(3,16));self.assertLessEqual(a*a,q);self.assertGreaterEqual(b*b,q)
    def test_direction_global_sign(self):
        t=pair(phase=1j);a=integrated_finite_law(t,[(1,0),(F(1,2),F(1,2))],F(1,4));b=integrated_finite_law(t,[(-1,0),(-F(1,2),-F(1,2))],F(1,4));self.assertEqual(a['intervals'],b['intervals'])
    def test_single_mode_compatibility(self):
        t=np.array([1,0,F(1,2),0],complex).reshape(1,4,1);a=integrated_finite_law([t],[(1,0)],F(1,4))['intervals'];b=integrated_vacuum(t,F(1,4))['law']
        for x,y in zip(a,b):self.assertLessEqual(max(x[0],y[0]),min(x[1],y[1]))
    def test_correlated_covariance(self):
        r=gaussian_rank_one([[0,0],[0,0]],[[0,0],[0,0]],[(1,0),(0,1)],F(1,4));self.assertEqual(r['covariance'][0][3],F(1,4));self.assertEqual(r['covariance'][0][0],F(5,4));self.assertEqual(r['generator_variance'],2)
    def test_common_opposite_generator(self):
        B=[[0,F(1,2)],[F(1,2),0]];a=gaussian_rank_one(B,[[0,0],[0,0]],[(1,0),(1,0)],F(1,4));b=gaussian_rank_one(B,[[0,0],[0,0]],[(1,0),(-1,0)],F(1,4));self.assertEqual(a['generator_variance'],F(2,3));self.assertEqual(b['generator_variance'],6)
    def test_coupling_scale_identity(self):
        r=gaussian_rank_one([[0]],[[0]],[(1,0)],F(1,4));g=growing_grid(F(1,4),16);self.assertEqual(coupling_bound(r,g),(1-g['tail_interval'][0])*g['interior_half_width']+g['tail_mean_excess_interval'][1])
    def test_invalid_direction_and_sigma(self):
        with self.assertRaises(ValueError):integrated_finite_law(pair(),[(0,0),(0,0)],F(1,4))
        with self.assertRaises(ValueError):integrated_finite_law(pair(),[(1,0),(1,0)],1)
    def test_nonphysical_pairing_rejected(self):
        with self.assertRaises(ValueError):gaussian_rank_one([[0,2],[2,0]],[[0,0],[0,0]],[(1,0),(1,0)],F(1,4))
    def test_zero_state_rejected(self):
        with self.assertRaises(ValueError):integrated_finite_law([np.zeros((1,2,1),complex)],[(1,0)],F(1,4))
