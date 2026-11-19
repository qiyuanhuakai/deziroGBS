import unittest
from fractions import Fraction as F
import numpy as np
from gbslab.certified_gaussian import HermitianCertificate,ExactPairingModel,CertificationFailure,certify_uncertainty,sqrt_interval,exact_uncertainty_rayleigh
from gbslab.banded_gaussian_mps import pairing_covariance


class CertifiedGaussianTests(unittest.TestCase):
    def test_complex_exact_determinant(self):
        c=HermitianCertificate([[2,F(1,4)],[F(1,4),3]],[[0,F(1,8)],[-F(1,8),0]])
        lo,hi=c.determinant_interval();self.assertLessEqual(lo,F(379,64));self.assertGreaterEqual(hi,F(379,64));self.assertLess(hi-lo,F(1,1<<110))

    def test_exact_sqrt(self):
        for x in (F(0),F(2),F(9,16),F(1,1<<200)):
            a,b=sqrt_interval(x,x);self.assertLessEqual(a*a,x);self.assertGreaterEqual(b*b,x);self.assertLessEqual(b-a,F(1,1<<256))

    def test_indefinite_rejected(self):
        with self.assertRaises(CertificationFailure):HermitianCertificate([[1,2],[2,1]])

    def test_all_nondyadic_entries_rejected(self):
        with self.assertRaises(ValueError):HermitianCertificate([[F(1,4),F(1,3)],[F(1,3),1]])

    def test_boundary_unresolved(self):
        self.assertEqual(certify_uncertainty(np.eye(4))['status'],'unresolved')

    def test_near_boundary_strict(self):
        self.assertEqual(certify_uncertainty((1+2.**-40)*np.eye(4))['status'],'strictly_physical')

    def test_negative_exact_witness(self):
        r=certify_uncertainty(.5*np.eye(4));self.assertEqual(r['status'],'strict_violation');self.assertEqual(r['exact_rayleigh'],F(-1,2))

    def test_one_mode_pairing(self):
        model=ExactPairingModel([[.25]],np.zeros((2,0)));V=np.diag([5/3,3/5]);a,b=model.vacuum_interval;self.assertLessEqual(a*a,F(15,16));self.assertGreaterEqual(b*b,F(15,16));self.assertLess(np.max(abs(V-model.covariance_center_float)),1e-15)

    def test_complex_cayley_convention(self):
        B=np.array([[.125+.0625j,.03125j],[.03125j,-.0625+.125j]]);noise=np.array([[.125],[-.125],[.25],[0]])
        model=ExactPairingModel(B,noise);V=pairing_covariance(B)+noise@noise.T;self.assertLess(np.max(abs(V-model.covariance_center_float)),1e-14);self.assertLess(float(model.compare_covariance(V)['operator_difference_upper']),1e-13)

    def test_transfer_to_explicit_noise(self):
        model=ExactPairingModel(np.zeros((2,2)),np.zeros((4,0)));V=np.eye(4);shifted=V+2.**-20*np.eye(4);r=model.noise_transfer_bound(V,shifted);self.assertTrue(r['noise_order_proved']);self.assertEqual(r['all_measurement_TV_transfer_upper'],F(1,1024));self.assertIsNone(model.noise_transfer_bound(V,V)['all_measurement_TV_transfer_upper'])


if __name__=='__main__':unittest.main()
