import unittest
from decimal import Decimal,localcontext
from fractions import Fraction as F
import numpy as np
from gbslab.displaced_fock_certificate import exp_minus_interval,displaced_dimension_two_law

class DisplacedCertificateTests(unittest.TestCase):
    def one(self,amplitudes):return [np.array(amplitudes,complex).reshape(1,2,1)]
    def bell(self,phase=1):
        a=np.zeros((1,2,2),complex);b=np.zeros((2,2,1),complex);a[0,0,0]=1;a[0,1,1]=phase;b[0,0,0]=b[1,1,0]=1;return [a,b]
    def test_exponential_zero(self):self.assertEqual(exp_minus_interval(0),(1,1))
    def test_exponential_decimal_crosscheck(self):
        with localcontext() as context:
            context.prec=400
            for x in (1,2,4):
                lo,hi=exp_minus_interval(x);reference=F((-Decimal(x)).exp());self.assertLess(lo,reference);self.assertLess(reference,hi)
    def test_exponential_guards(self):
        for x,k in [(-1,160),(5,160),(1,15),(1,161),(1,True)]:
            with self.assertRaises(ValueError):exp_minus_interval(x,k)
    def test_vacuum_nonzero_displacement(self):
        r=displaced_dimension_two_law(self.one([1,0]),[.5j]);self.assertEqual(tuple(r['intervals'][0]),exp_minus_interval(F(1,4)))
    def test_single_photon_exact_formula(self):
        r=displaced_dimension_two_law(self.one([0,1]),[.5+.25j]);v=r['vacuum_records'][1];self.assertEqual(v['coefficient'],F(5,16));self.assertEqual(v['exponent'],F(5,16))
    def test_click_includes_unbounded_output(self):
        r=displaced_dimension_two_law(self.one([0,1]),[1]);lo,hi=exp_minus_interval(1);self.assertEqual(tuple(r['intervals'][1]),(1-hi,1-lo))
    def test_bell_interference_polynomial(self):
        alpha=.5+.25j;beta=.25-.5j;r=displaced_dimension_two_law(self.bell(1j),[alpha,beta]);z=1+1j*alpha.conjugate()*beta.conjugate();expected=(F(z.real)**2+F(z.imag)**2)/2;self.assertEqual(r['vacuum_records'][3]['coefficient'],expected)
    def test_exact_global_scale_invariance(self):
        a=self.bell();r=displaced_dimension_two_law(a,[.25,.5j]);a[0]*=2j;self.assertEqual(r['intervals'],displaced_dimension_two_law(a,[.25,.5j])['intervals'])
    def test_zero_displacement_bell(self):
        self.assertEqual(displaced_dimension_two_law(self.bell(),[0,0])['intervals'],[(F(1,2),F(1,2)),(0,0),(0,0),(F(1,2),F(1,2))])
    def test_zero_state_refused(self):
        with self.assertRaises(ArithmeticError):displaced_dimension_two_law(self.one([0,0]),[.5])
    def test_dimension_three_refused(self):
        with self.assertRaises(ValueError):displaced_dimension_two_law([np.ones((1,3,1),complex)],[0])
