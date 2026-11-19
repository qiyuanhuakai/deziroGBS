import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.gaussian_fock_bridge import root_interval,physical_pairing,hermite_box,gaussian_fidelity,gaussian_displaced_law


class GaussianFockBridgeTests(unittest.TestCase):
    def test_root_rational_enclosure(self):
        for x in (F(0),F(1,9),F(7,13),F(625)):
            a,b=root_interval(x);self.assertTrue(a*a<=x<=b*b);self.assertLessEqual(b-a,F(1,1<<192))
    def test_invalid_pairing_rejected(self):
        for r,i in (([[1]],[[0]]),([[2]],[[0]]),([[0,1],[0,0]],[[0,0],[0,0]])):
            with self.assertRaises(ValueError):physical_pairing(r,i)
    def test_single_mode_hermite(self):
        _,H=hermite_box([[F(1,2)]],[[F(0)]],[8]);self.assertEqual(H,[(F(1),0),(0,0),(F(1,2),0),(0,0),(F(3,4),0),(0,0),(F(15,8),0),(0,0)])
    def test_three_mode_product(self):
        r=[[F(i==j,2) for j in range(3)] for i in range(3)];i=[[F(0)]*3 for _ in range(3)];o,H=hermite_box(r,i,[3]*3);self.assertEqual(H[o.index((2,2,2))],(F(1,8),F(0)));self.assertEqual(H[o.index((1,1,0))],(0,0))
    def test_vacuum_fidelity_includes_gaussian_tail(self):
        t=[np.array([1,0,0,0],complex).reshape(1,4,1)];r=gaussian_fidelity(t,[[F(1,2)]],[[0]]);lo,hi=r['full_fidelity_interval'];self.assertTrue(lo*lo<=F(3,4)<=hi*hi);self.assertGreater(r['tail_mass_interval'][0],0);self.assertGreater(r['compression_trace_interval'][0],0)
    def test_global_scale_and_phase_invariance(self):
        t=np.array([1,.25j,.125,0],complex).reshape(1,4,1);a=gaussian_fidelity([t],[[F(1,2)]],[[0]]);b=gaussian_fidelity([2j*t],[[F(1,2)]],[[0]]);self.assertEqual(a['full_fidelity_interval'],b['full_fidelity_interval'])
    def test_zero_gaussian_is_exact_vacuum(self):
        t=[np.ones((1,1,1),complex)];r=gaussian_fidelity(t,[[0]],[[0]]);self.assertEqual(r['full_trace_interval'],(0,0));p=gaussian_displaced_law([[0]],[[0]],[0]);self.assertEqual(p['intervals'],[(1,1),(0,0)])
    def test_displaced_squeezed_vacuum_formula(self):
        p=gaussian_displaced_law([[F(1,2)]],[[0]],[.5]);self.assertEqual(p['records'][1]['exponent'],F(1,8));self.assertEqual(p['records'][1]['determinant'],F(4,3))
    def test_resource_and_displacement_guards(self):
        with self.assertRaises(ValueError):hermite_box([[F(0)]],[[F(0)]],[9])
        with self.assertRaises(ValueError):gaussian_displaced_law([[0]],[[0]],[np.nan])
        with self.assertRaises(ValueError):root_interval(-1)

if __name__=='__main__':unittest.main()
