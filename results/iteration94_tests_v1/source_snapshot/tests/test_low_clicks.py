import math
import unittest
import mpmath as mp
import numpy as np
from gbslab.low_clicks import gaussian_low_clicks,mps_low_clicks,low_click_tv
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.gaussian import haar_unitary
from gbslab.exact import ThresholdModel
from gbslab.mps_probe import dense_to_mps,mps_to_dense
from test_banded_gaussian_mps import reference_box


def indices(patterns):
    return [sum(1<<int(i) for i in row if i>=0) for row in patterns]


def threshold_dense(state):
    p=np.zeros(1<<state.ndim)
    for n in np.ndindex(state.shape): p[sum(1<<i for i,k in enumerate(n) if k)]+=abs(state[n])**2
    return p


class LowClickTests(unittest.TestCase):
    def test_analytic_vacuum_product_and_pair(self):
        vacuum=gaussian_low_clicks(np.zeros((4,4))); np.testing.assert_array_equal(vacuum['probabilities'],[1]+[0]*10)
        B=np.diag([.2,.3j]); r=gaussian_low_clicks(B); z=np.sqrt(1-np.abs(np.diag(B))**2)
        np.testing.assert_allclose(r['probabilities'],[z.prod(),(1-z[0])*z[1],z[0]*(1-z[1]),(1-z[0])*(1-z[1])],atol=1e-16,rtol=1e-13)
        r=gaussian_low_clicks(np.array([[0,.3j],[.3j,0]])); np.testing.assert_allclose(r['probabilities'],[.91,0,0,.09],atol=1e-15,rtol=0)

    def test_all_small_gaussian_events_against_covariance_inclusion_exclusion(self):
        for m in (3,4,6):
            U=haar_unitary(m,np.random.default_rng(260100+m)); B=-(U*np.tanh(np.linspace(.05,.55,m)))@U.T
            r=gaussian_low_clicks(B); full=ThresholdModel(pairing_covariance(B)).full_distribution()
            np.testing.assert_allclose(r['probabilities'],full[indices(r['patterns'])],atol=2e-13,rtol=0)

    def test_finite_mps_all_occupations_and_gauges(self):
        rng=np.random.default_rng(260201); dense=rng.normal(size=(4,)*4)+1j*rng.normal(size=(4,)*4); dense/=np.linalg.norm(dense)
        tensors,_=dense_to_mps(dense,16); tensors[0]*=(2+3j)
        r=mps_low_clicks(tensors); full=threshold_dense(dense)
        np.testing.assert_allclose(r['probabilities'],full[indices(r['patterns'])],atol=1e-14,rtol=0)
        self.assertAlmostEqual(r['input_log_norm_squared'],math.log(13),delta=1e-13)

    def test_coarse_bound_below_full_tv_and_all_nonzero_occupations(self):
        U=haar_unitary(4,np.random.default_rng(260202)); B=-(U*np.tanh([.1,.2,.3,.4]))@U.T
        box=reference_box(B,4); tensors,_=dense_to_mps(box/np.linalg.norm(box),4)
        a=mps_low_clicks(tensors); b=gaussian_low_clicks(B); bound=low_click_tv(a,b)
        full_a=threshold_dense(mps_to_dense(tensors)); full_b=ThresholdModel(pairing_covariance(B)).full_distribution()
        self.assertLessEqual(bound['threshold_coarse_tv_lower_estimate'],.5*np.sum(abs(full_a-full_b))+1e-12)
        self.assertGreaterEqual(bound['threshold_coarse_tv_lower_estimate'],bound['vacuum_probability_gap']-1e-14)
        # Four photons at two sites contribute to a two-click pattern.
        dense=np.zeros((4,4),complex); dense[3,1]=1; tensors,_=dense_to_mps(dense,4)
        np.testing.assert_allclose(mps_low_clicks(tensors)['probabilities'],[0,0,0,1],atol=1e-15)

    def test_weak_edges_against_high_precision_determinant(self):
        for a,b,c in ((0j,0j,1e-50j),(.2+.1j,-.3j,1e-20),(.4,.4j,1e-10j),(.8,-.8,.01j)):
            B=np.array([[a,c],[c,b]],complex); result=gaussian_low_clicks(B)['probabilities'][-1]
            with mp.workdps(180):
                A=mp.matrix([[mp.mpc(a),mp.mpc(c)],[mp.mpc(c),mp.mpc(b)]]); determinant=mp.det(mp.eye(2)-A*A.H)
                p0=mp.sqrt(determinant.real); expected=(1-p0/mp.sqrt(1-abs(mp.mpc(a))**2)-p0/mp.sqrt(1-abs(mp.mpc(b))**2)+p0)
                self.assertLess(abs(result/float(expected)-1),2e-13)
            self.assertGreater(result,0)

    def test_invalid_inputs_and_explicit_positive_underflow(self):
        with self.assertRaises(ValueError): gaussian_low_clicks(np.eye(2))
        with self.assertRaises(ArithmeticError): gaussian_low_clicks(np.array([[0,1e-200],[1e-200,0]]))
        with self.assertRaises(ArithmeticError): mps_low_clicks([np.array([[[1.],[1e-200]]])])
        a=gaussian_low_clicks(np.zeros((2,2))); b=gaussian_low_clicks(np.zeros((3,3)))
        with self.assertRaises(ValueError): low_click_tv(a,b)


if __name__=='__main__': unittest.main()
