import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.displaced_fock_interval import reciprocal_root_integer,displaced_finite_law
from gbslab.displaced_fock_certificate import displaced_dimension_two_law

class FiniteDisplacementTests(unittest.TestCase):
    def one(self,n):
        a=np.zeros((1,n+1,1),complex);a[0,n,0]=1;return [a]
    def test_root_exact_and_irrational(self):
        self.assertEqual(reciprocal_root_integer(4),(F(1,2),F(1,2)));a,b=reciprocal_root_integer(2);self.assertLess(a*a*2,1);self.assertGreater(b*b*2,1)
    def test_root_guards(self):
        for n,b in [(0,192),(-1,192),(2,12),(2,513),(True,192)]:
            with self.assertRaises(ValueError):reciprocal_root_integer(n,b)
    def test_number_state_polynomial(self):
        for n,c in [(2,F(25,512)),(3,F(125,24576))]:
            r=displaced_finite_law(self.one(n),[.5+.25j]);a,b=r['vacuum_records'][1]['coefficient_interval'];self.assertLessEqual(a,c);self.assertLessEqual(c,b)
    def test_noon_interference(self):
        a=np.zeros((1,3,2),complex);b=np.zeros((2,3,1),complex);a[0,2,0]=b[0,0,0]=b[1,2,0]=1;a[0,0,1]=1j;alpha=.5+.25j;beta=.25-.5j;r=displaced_finite_law([a,b],[alpha,beta]);z=alpha.conjugate()**2+1j*beta.conjugate()**2;c=(F(z.real)**2+F(z.imag)**2)/4;lo,hi=r['vacuum_records'][3]['coefficient_interval'];self.assertLessEqual(lo,c);self.assertLessEqual(c,hi)
    def test_dimension_two_agrees(self):
        t=[np.array([1,.5j],complex).reshape(1,2,1)];alpha=[.25+.5j];self.assertEqual(displaced_finite_law(t,alpha)['intervals'],displaced_dimension_two_law(t,alpha)['intervals'])
    def test_global_complex_scaling(self):
        t=self.one(3);r=displaced_finite_law(t,[.5]);t[0]*=2j;self.assertEqual(r['intervals'],displaced_finite_law(t,[.5])['intervals'])
    def test_fock_box_cap(self):
        with self.assertRaises(MemoryError):displaced_finite_law(self.one(3),[.5],max_states=3)
