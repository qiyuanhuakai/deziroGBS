import math
import unittest
import numpy as np
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.gaussian_schmidt_mps import williamson,gaussian_unitary_kernel,apply_gaussian_kernel,top_occupations,projected_pair_tensor,build_schmidt_core
from gbslab.gaussian_mps_feasibility import equal_core_spectra
from gbslab.mps_probe import mps_to_dense,geometric_top_weights
from test_banded_gaussian_mps import reference_box


class GaussianSchmidtMpsTests(unittest.TestCase):
    def test_williamson_mixed_covariance_and_purity(self):
        rng=np.random.default_rng(2101)
        for m in (1,2,5):
            u=haar_unitary(m,rng); O=np.block([[u.real,-u.imag],[u.imag,u.real]])
            squeeze=np.exp(np.r_[np.arange(m)*.1,-np.arange(m)*.1]); thermal=1+np.arange(m)*.2
            V=(O*(squeeze**2*np.r_[thermal,thermal]))@O.T
            S,nu,errors=williamson(V)
            np.testing.assert_allclose(np.sort(nu),thermal,atol=1e-13,rtol=0)
            self.assertLess(errors['symplectic_residual'],1e-13)
            np.testing.assert_allclose((S*np.r_[nu,nu])@S.T,V,atol=1e-13,rtol=0)

    def test_unitary_kernel_inverse_and_scalar(self):
        rng=np.random.default_rng(2102)
        for m in (2,4,6):
            u=haar_unitary(m,rng); B=(u*(-np.tanh(np.linspace(.1,.5,m))))@u.T
            V=pairing_covariance(B); l=m//2; idx=list(range(l))+list(range(m,m+l))
            S,_,_=williamson(V[np.ix_(idx,idx)]); K,c=gaussian_unitary_kernel(S)
            order=list(range(l,2*l))+list(range(l)); inverse=K[np.ix_(order,order)].conj()
            transformed,f=apply_gaussian_kernel(B,inverse,l); recovered,g=apply_gaussian_kernel(transformed,K,l)
            np.testing.assert_allclose(recovered,B,atol=1e-13,rtol=0)
            self.assertAlmostEqual(abs(c*c*f*g-1),0,delta=1e-13)

    def test_heap_basis_and_downward_closure(self):
        q=np.array([.2,0,.1,.2]); states,weights=top_occupations(q,50)
        np.testing.assert_allclose(weights,geometric_top_weights(q,50),atol=1e-16,rtol=0)
        lookup={tuple(n) for n in states}
        for n in states:
            for j in np.flatnonzero(n):
                sub=n.copy(); sub[j]-=1; self.assertIn(tuple(sub),lookup)
        self.assertTrue(np.all(states[:,1]==0))

    def test_projected_creations_against_independent_recurrence(self):
        rng=np.random.default_rng(2103); u=haar_unitary(5,rng); B=-.3*(u@u.T)
        left,_=top_occupations(np.array([.2,.1]),9); right,_=top_occupations(np.array([.15,.12]),11)
        actual,_=projected_pair_tensor(B,left,4,right)
        d=max(4,int(left.max())+1,int(right.max())+1); reference=reference_box(B,d)/reference_box(np.zeros((1,1)),1)[0]
        from gbslab.banded_gaussian_mps import pairing_log_normalization
        reference/=math.exp(pairing_log_normalization(B))
        expected=np.empty_like(actual)
        for a,l in enumerate(left):
            for p in range(4):
                for b,r in enumerate(right): expected[a,p,b]=reference[tuple(l)+(p,)+tuple(r)]
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)
        with self.assertRaises(ValueError): projected_pair_tensor(np.zeros((2,2)),np.array([[0],[2]]),3,np.zeros((1,0),int))

    def test_single_squeeze_kernel_and_identity(self):
        for s in (0.,.3):
            K,c=gaussian_unitary_kernel(np.diag([math.exp(-s),math.exp(s)]))
            A,_=projected_pair_tensor(K,np.zeros((1,0),int),8,np.arange(8)[:,None]); A=A[0]*c
            if s==0: np.testing.assert_allclose(A,np.eye(8),atol=1e-15)
            expected=np.zeros(8)
            for n in range(0,8,2): expected[n]=(-math.tanh(s))**(n//2)*math.sqrt(math.factorial(n))/(2**(n//2)*math.factorial(n//2)*math.sqrt(math.cosh(s)))
            np.testing.assert_allclose(A[:,0],expected,atol=1e-15)

    def test_original_cut_spectra_and_small_state_bound(self):
        for m in (2,4,6):
            u=haar_unitary(m,np.random.default_rng(2104+m)); B=-math.tanh(.35)*(u@u.T)
            tensors,info=build_schmidt_core(B,4,8); state=mps_to_dense(tensors); ref=reference_box(B,4)
            actual=math.sqrt(max(0.,1-abs(np.vdot(ref,state))**2))
            self.assertLessEqual(actual,info['construction_trace_budget']+1e-11)
            spectra=equal_core_spectra(u,.35,8)
            for record,cut in zip(info['cuts'],spectra['cuts']):
                np.testing.assert_allclose(geometric_top_weights(record['parameters'],8),cut['top_weights'],atol=2e-13,rtol=0)

    def test_original_projector_product_independent_dense(self):
        # Full Gaussian Schmidt projectors are approximated independently in a
        # much larger physical box; its omitted mass is negligible here.
        m=3; dref=12; d=4; rank=2; u=haar_unitary(m,np.random.default_rng(2105)); B=-math.tanh(.18)*(u@u.T)
        reference=reference_box(B,dref); self.assertLess(1-np.linalg.norm(reference)**2,1e-9)
        projected=reference.copy()
        for cut in range(m-1,0,-1):
            left,_,_=np.linalg.svd(reference.reshape(dref**cut,-1),full_matrices=False); kept=left[:,:rank]
            projected=(kept@(kept.conj().T@projected.reshape(dref**cut,-1))).reshape(reference.shape)
        expected=projected[(slice(0,d),)*m]; expected/=np.linalg.norm(expected)
        tensors,info=build_schmidt_core(B,d,rank); actual=mps_to_dense(tensors); overlap=np.vdot(expected,actual); actual/=overlap/abs(overlap)
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=0)

    def test_vacuum_pair_state_and_resource_guard(self):
        vacuum,info=build_schmidt_core(np.zeros((4,4)),4,8)
        self.assertEqual(info['construction_trace_budget'],0); self.assertEqual(info['projection_success'],1)
        B=np.array([[0,.3],[.3,0]],complex); tensors,info=build_schmidt_core(B,6,6); state=mps_to_dense(tensors)
        expected=np.diag(.3**np.arange(6)); expected/=np.linalg.norm(expected)
        overlap=np.vdot(expected,state); np.testing.assert_allclose(state/(overlap/abs(overlap)),expected,atol=2e-14)
        with self.assertRaises(MemoryError): projected_pair_tensor(np.zeros((3,3)),np.arange(4)[:,None],4,np.arange(4)[:,None],max_elements=10)


if __name__=='__main__': unittest.main()
