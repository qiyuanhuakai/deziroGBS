import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.real_noise_integral import integrated_vacuum


class RealNoiseIntegralTests(unittest.TestCase):
    def test_vacuum_closed_gaussian_integral(self):
        r=integrated_vacuum(np.array([1,0,0],complex).reshape(1,3,1),F(1,2));a,b=r['vacuum_interval'];self.assertTrue(a*a<=F(2,3)<=b*b)
    def test_single_photon_integral(self):
        r=integrated_vacuum(np.array([0,1,0],complex).reshape(1,3,1),F(1,2));a,b=r['vacuum_interval'];self.assertTrue(a*a<=F(1,54)<=b*b)
    def test_zero_noise_is_vacuum_projection(self):
        r=integrated_vacuum(np.array([1,1j],complex).reshape(1,2,1),0);self.assertEqual(r['vacuum_interval'],(F(1,2),F(1,2)))
    def test_global_phase_and_scale(self):
        t=np.array([1,.5j,.25],complex).reshape(1,3,1);a=integrated_vacuum(t,F(1,4));b=integrated_vacuum(2j*t,F(1,4));self.assertEqual(a['vacuum_interval'],b['vacuum_interval'])
    def test_odd_interference_integrates_to_zero(self):
        t=np.array([1,1],complex).reshape(1,2,1);a=integrated_vacuum(t,F(1,4));b=integrated_vacuum(t*np.array([1,-1]).reshape(1,2,1),F(1,4));self.assertEqual(a['vacuum_interval'],b['vacuum_interval'])
    def test_guards(self):
        with self.assertRaises(ValueError):integrated_vacuum(np.zeros((1,2,1)),F(1,4))
        with self.assertRaises(ValueError):integrated_vacuum(np.ones((1,9,1)),F(1,4))
        with self.assertRaises(ValueError):integrated_vacuum(np.ones((1,2,1)),-1)

if __name__=='__main__':unittest.main()
