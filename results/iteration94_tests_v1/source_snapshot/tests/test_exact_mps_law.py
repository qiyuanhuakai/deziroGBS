import unittest
import numpy as np
from fractions import Fraction as F
from gbslab.exact_mps_law import represented_threshold_law

class ExactMPSLawTests(unittest.TestCase):
    def chain(self):
        a=np.zeros((1,2,2),complex);b=np.zeros((2,2,1),complex);a[0,0,0]=1;a[0,1,1]=1j;b[0,0,0]=1;b[1,1,0]=2
        return [a,b]
    def test_complex_entangled_known_law(self):
        r=represented_threshold_law(self.chain());self.assertEqual(r['distribution'],[F(1,5),0,0,F(4,5)])
    def test_dyadic_gauge_invariance(self):
        a,b=self.chain();ref=represented_threshold_law([a,b]);a[:,:,0]*=4;b[0,:,:]/=4;a[:,:,1]*=.5;b[1,:,:]*=2
        self.assertEqual(ref['distribution'],represented_threshold_law([a,b])['distribution'])
    def test_global_complex_scale(self):
        a,b=self.chain();a*=2j;self.assertEqual(represented_threshold_law([a,b])['represented_norm'],20)
    def test_multiple_click_occupations(self):
        a=np.array([1,2j,3],complex).reshape(1,3,1);self.assertEqual(represented_threshold_law([a])['distribution'],[F(1,14),F(13,14)])
    def test_state_cap(self):
        with self.assertRaises(MemoryError):represented_threshold_law(self.chain(),max_states=3)
    def test_invalid_nonfinite(self):
        a,b=self.chain();a[0,0,0]=np.nan
        with self.assertRaises(ValueError):represented_threshold_law([a,b])
    def test_zero_state(self):
        a,b=self.chain();a*=0
        with self.assertRaises((ValueError,ArithmeticError)):represented_threshold_law([a,b])
