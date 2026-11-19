import math
import unittest
import numpy as np
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.mps_overlap import compare_mps
from gbslab.mps_probe import mps_to_dense
from test_banded_gaussian_mps import reference_box


class SchmidtRecurrenceBackendTests(unittest.TestCase):
    def test_same_projectors_and_states(self):
        for m in (2,4,6):
            u=haar_unitary(m,np.random.default_rng(230100+m)); B=-math.tanh(.38)*(u@u.T)
            for rank in (2,8):
                p,pi,pm=build_schmidt_core(B,4,rank,return_maps=True)
                r,ri,rm=build_schmidt_core(B,4,rank,return_maps=True,coefficient_backend='recurrence')
                self.assertLess(compare_mps(p,r)['pure_trace_distance_from_difference'],1e-12)
                self.assertAlmostEqual(pi['projection_success'],ri['projection_success'],delta=1e-13)
                self.assertEqual(pi['construction_trace_budget'],ri['construction_trace_budget'])
                for a,b in zip(pm['bases'],rm['bases']): np.testing.assert_array_equal(a,b)
                for a,b in zip(pm['kernels'],rm['kernels']): np.testing.assert_array_equal(a,b)

    def test_original_gaussian_quality(self):
        for m in (3,5):
            u=haar_unitary(m,np.random.default_rng(230200+m)); B=-.28*(u@u.T)
            tensors,info=build_schmidt_core(B,5,8,coefficient_backend='recurrence')
            reference=reference_box(B,5); state=mps_to_dense(tensors)
            distance=math.sqrt(max(0.,1-abs(np.vdot(reference,state))**2))
            self.assertLessEqual(distance,info['construction_trace_budget']+1e-12)

    def test_original_projectors_dense(self):
        m=3; dref=12; d=4; rank=2
        u=haar_unitary(m,np.random.default_rng(230301)); B=-math.tanh(.18)*(u@u.T)
        reference=reference_box(B,dref); projected=reference.copy()
        for cut in range(m-1,0,-1):
            left,_,_=np.linalg.svd(reference.reshape(dref**cut,-1),full_matrices=False); keep=left[:,:rank]
            projected=(keep@(keep.conj().T@projected.reshape(dref**cut,-1))).reshape(reference.shape)
        expected=projected[(slice(0,d),)*m]; expected/=np.linalg.norm(expected)
        tensors,_=build_schmidt_core(B,d,rank,coefficient_backend='recurrence'); actual=mps_to_dense(tensors)
        phase=np.vdot(expected,actual); actual*=phase.conjugate()/abs(phase)
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=0)

    def test_vacuum_and_pair(self):
        tensors,info=build_schmidt_core(np.zeros((4,4)),4,8,coefficient_backend='recurrence')
        self.assertEqual(info['projection_success'],1); self.assertEqual(info['construction_trace_budget'],0)
        B=np.array([[0,.3j],[.3j,0]])
        tensors,_=build_schmidt_core(B,6,6,coefficient_backend='recurrence')
        expected=np.diag((.3j)**np.arange(6)); expected/=np.linalg.norm(expected)
        actual=mps_to_dense(tensors); phase=np.vdot(expected,actual)
        np.testing.assert_allclose(actual*phase.conjugate()/abs(phase),expected,atol=2e-14,rtol=0)

    def test_rejection_and_default(self):
        B=np.array([[0,.2],[.2,0]],complex)
        with self.assertRaises(ValueError): build_schmidt_core(B,4,4,coefficient_backend='unknown')
        with self.assertRaises(MemoryError): build_schmidt_core(B,4,4,coefficient_backend='recurrence',max_elements=1)
        with self.assertRaises(TimeoutError): build_schmidt_core(B,4,4,coefficient_backend='recurrence',max_seconds=-1)
        a,_=build_schmidt_core(B,4,4); b,_=build_schmidt_core(B,4,4,coefficient_backend='pco')
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)


if __name__=='__main__': unittest.main()
