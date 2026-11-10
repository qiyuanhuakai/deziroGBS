import unittest
import numpy as np
from gbslab.gaussian_resource import weighted_pure,resource_diagnostics,search_resource,guarded_resource,passive_active_covariance,pure_pairing,omega
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.gaussian_schmidt_mps import williamson
from gbslab.banded_gaussian_mps import pairing_covariance


class GaussianResourceTests(unittest.TestCase):
    def test_weighted_oracle_analytic_and_riccati(self):
        A=np.diag([2.,5.,3.,7.]); P,value=weighted_pure(A)
        expected=np.diag(np.r_[np.sqrt([3/2,7/5]),np.sqrt([2/3,5/7])])
        np.testing.assert_allclose(P,expected,atol=2e-15,rtol=0)
        self.assertAlmostEqual(value,2*(np.sqrt(6)+np.sqrt(35)),delta=1e-13)
        rng=np.random.default_rng(281201); X=rng.normal(size=(8,8)); A=np.eye(8)+X@X.T; P,value=weighted_pure(A); O=omega(4)
        np.testing.assert_allclose(P@O@P,O,atol=2e-13,rtol=0)
        np.testing.assert_allclose(P@A@P,O.T@A@O,atol=3e-12,rtol=0)

    def test_oracle_independent_williamson(self):
        rng=np.random.default_rng(281202); X=rng.normal(size=(8,8)); A=np.eye(8)+X@X.T
        S,nu,_=williamson(A); inverse=np.linalg.inv(S); expected=inverse.T@inverse
        P,value=weighted_pure(A); np.testing.assert_allclose(P,expected,atol=3e-13,rtol=0)
        self.assertAlmostEqual(value,2*sum(nu),delta=1e-11)

    def test_envelope_gradient_finite_difference(self):
        rng=np.random.default_rng(281203); L=rng.normal(size=(6,6))*.2; E=rng.normal(size=(6,6)); E/=np.linalg.norm(E); V=1.5*np.eye(6)
        def f(L):
            S=L@L.T; P,value=weighted_pure(np.eye(6)+S)
            return np.sum(S*V.T)-value,2*(V-P)@L
        _,G=f(L); h=1e-4; finite=(f(L+h*E)[0]-f(L-h*E)[0])/(2*h)
        self.assertAlmostEqual(finite,float(np.sum(G*E)),delta=3e-8)

    def test_guarded_known_passive_solution(self):
        U=haar_unitary(3,np.random.default_rng(281204)); V=covariance_from_interferometer(U,[.2,.5,.8],.6)
        r=guarded_resource(V,margin=1e-5); self.assertTrue(r['numerically_accepted'])
        squeezing=.6*np.exp(-2*np.array([.2,.5,.8]))+.4
        expected=float(np.sum(squeezing+1/squeezing-2)/4)
        self.assertLess(abs(r['diagnostics']['pure_mean_photons']-expected),2e-5)
        self.assertGreater(r['diagnostics']['minimum_remainder_eigenvalue'],1e-6)
        np.testing.assert_allclose(r['noise_factor']@r['noise_factor'].T,r['remainder'],atol=3e-15,rtol=0)
        self.assertLessEqual(r['diagnostics']['dual_photon_lower_estimate'],expected+1e-11)

    def test_unsuccessful_saved_candidate_retained_without_noise(self):
        V=covariance_from_interferometer(haar_unitary(4,np.random.default_rng(281205)),[.2,.4,.6,.8],[.2,.4,.6,.8])
        r=guarded_resource(V,margin=1e-6,max_iterations=1)
        self.assertFalse(r['numerically_accepted']); self.assertIsNone(r['noise_factor'])
        np.testing.assert_array_equal(r['remainder'],V-r['pure_covariance'])

    def test_passive_support_compression_and_error_refusal(self):
        U=haar_unitary(5,np.random.default_rng(281206)); V=covariance_from_interferometer(U,[.2,.4,0,0,0],.7)
        r=passive_active_covariance(V); self.assertEqual(r['active_modes'],2)
        self.assertLess(r['operator_error'],2e-14); O=r['basis']
        np.testing.assert_allclose(O.T@omega(5)@O,omega(2),atol=3e-15,rtol=0)
        with self.assertRaises(ArithmeticError): passive_active_covariance(V,eigen_threshold=10)
        vacuum=passive_active_covariance(np.eye(6)); self.assertEqual(vacuum['active_modes'],0)

    def test_pure_encoding_round_trip(self):
        U=haar_unitary(4,np.random.default_rng(281207)); B=-(U*np.tanh([.1,.2,.3,.4]))@U.T; V=pairing_covariance(B)
        encoded,canonical,error=pure_pairing(V)
        np.testing.assert_allclose(encoded,B,atol=2e-15,rtol=0); self.assertLess(error,3e-15)
        with self.assertRaises(ValueError): pure_pairing(1.1*np.eye(4))

    def test_invalid_inputs_and_explicit_strict_margin(self):
        for A in (np.eye(3),np.diag([0.,1.]),np.eye(2)*(1+1j)):
            with self.assertRaises(ValueError): weighted_pure(A)
        with self.assertRaises(ValueError): guarded_resource(np.eye(4),margin=1e-6)
        with self.assertRaises(ValueError): search_resource(.9*np.eye(4))
        with self.assertRaises(ValueError): guarded_resource(1.2*np.eye(4),margin=-1)


if __name__=='__main__': unittest.main()
