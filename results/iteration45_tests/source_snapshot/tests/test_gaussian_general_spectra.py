import math
import unittest
import numpy as np
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_general_spectra import pairing_spectra,pairing_representation_bounds
from gbslab.gaussian_mps_feasibility import equal_core_spectra
from gbslab.mps_probe import dense_to_mps,mps_to_dense
from test_banded_gaussian_mps import reference_box


class GeneralGaussianSpectraTests(unittest.TestCase):
    def test_analytic_pair(self):
        B=np.array([[0,.3j],[.3j,0]]); result=pairing_spectra(B,16); cut=result['cuts'][0]
        np.testing.assert_allclose(cut['parameters'],[.09],atol=1e-14,rtol=0)
        np.testing.assert_allclose(cut['top_weights'],.91*.09**np.arange(16),atol=1e-14,rtol=0)

    def test_equal_core_independent_formula(self):
        for m in (4,9):
            U=haar_unitary(m,np.random.default_rng(250100+m)); B=-math.tanh(.6)*(U@U.T)
            observed=pairing_spectra(B,32); expected=equal_core_spectra(U,.6,32)
            for a,b in zip(observed['cuts'],expected['cuts']): np.testing.assert_allclose(a['top_weights'],b['top_weights'],atol=2e-13,rtol=0)

    def test_unequal_state_against_fock_svd_and_actual_distance(self):
        U=haar_unitary(3,np.random.default_rng(250201)); B=U@np.diag([.05,.07,.09])@U.T
        box=reference_box(B,10); spectrum=pairing_representation_bounds(B,dimensions=(10,),ranks=(2,16))
        for cut in spectrum['spectra']['cuts']:
            s=np.linalg.svd(box.reshape(10**cut['cut'],-1),compute_uv=False)**2
            np.testing.assert_allclose(s[:8],cut['top_weights'][:8],atol=2e-10,rtol=0)
        state,_=dense_to_mps(box/np.linalg.norm(box),2); actual=math.sqrt(max(0.,1-abs(np.vdot(box,mps_to_dense(state)))**2))
        lower=spectrum['rows'][0]['pure_rank_distance_lower_estimate']; self.assertLessEqual(lower,actual+1e-10)

    def test_product_squeezing_and_vacuum(self):
        for B in (np.zeros((4,4)),np.diag([.2,.4j,-.3,.1])):
            result=pairing_representation_bounds(B,ranks=(1,),dimensions=(4,))
            self.assertLess(result['rows'][0]['max_cut_tail'],2e-14)

    def test_input_guards(self):
        for B,r in ((np.eye(2),4),(np.zeros((1,1)),4),(np.zeros((2,2)),True),(np.zeros((2,2)),0)):
            with self.assertRaises(ValueError): pairing_spectra(B,r)
        with self.assertRaises(ValueError): pairing_representation_bounds(np.zeros((2,2)),ranks=())


if __name__=='__main__': unittest.main()
