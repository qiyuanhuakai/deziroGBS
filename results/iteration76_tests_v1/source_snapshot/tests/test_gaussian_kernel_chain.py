import math
import unittest
import numpy as np
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_kernel_chain import contract_gaussians,fold_kernel_chain,reverse_kernel_chain,GaussianContractionError
from gbslab.gaussian_schmidt_mps import decompose_gaussian
from gbslab.gaussian_pair_recurrence import pair_recurrence_tensor


class GaussianKernelChainTests(unittest.TestCase):
    def test_two_site_pair_and_disconnected_chain(self):
        Q=np.array([[0,.3j],[.3j,0]])
        B,c,info=fold_kernel_chain([Q,Q],[0,1,0])
        np.testing.assert_allclose(B,[[0,-.09],[-.09,0]],atol=1e-15)
        self.assertEqual(c,0); self.assertGreater(info['raw_log_norm_squared'],0)
        B,c,_=fold_kernel_chain([np.array([[.2]]),np.array([[.3j]])],[0,0,0])
        np.testing.assert_allclose(B,np.diag([.2,.3j])); self.assertEqual(c,0)

    def test_known_gaussian_decomposition_both_directions(self):
        for m in (2,3,6):
            U=haar_unitary(m,np.random.default_rng(240100+m)); original=-.3*(U@U.T)
            kernels,records,scalar,_=decompose_gaussian(original)
            scales=[math.log(r['kernel_scale']) for r in records]+[np.log(scalar)]
            bonds=list(range(m))+[0]
            B,c,info=fold_kernel_chain(kernels,bonds,log_scales=scales)
            kr,br,sr=reverse_kernel_chain(kernels,bonds,scales); C,d,ri=fold_kernel_chain(kr,br,log_scales=sr)
            np.testing.assert_allclose(B,original,atol=2e-13,rtol=0)
            np.testing.assert_allclose(C[::-1,::-1],B,atol=2e-13,rtol=0)
            self.assertLess(abs(np.exp(c-d)-1),2e-13)
            self.assertLess(abs(info['raw_log_norm_squared']),2e-12)
            self.assertLess(abs(ri['raw_log_norm_squared']),2e-12)

    def test_finite_virtual_sum_independent_reference(self):
        rng=np.random.default_rng(240201); ks=[]
        for size in (2,3,2):
            a=rng.normal(size=(size,size))+1j*rng.normal(size=(size,size)); ks.append(.035*(a+a.T))
        B,c,_=fold_kernel_chain(ks,[0,1,1,0]); empty=np.zeros((1,0),int); basis=np.arange(20)[:,None]
        ts=[pair_recurrence_tensor(ks[0],empty,4,basis)[0],pair_recurrence_tensor(ks[1],basis,4,basis)[0],pair_recurrence_tensor(ks[2],basis,4,empty)[0]]
        finite=np.einsum('api,iqj,jrb->pqr',*ts)
        # Generate physical coefficients with a separate grouping into left/right basis.
        left=np.array([(i,j) for i in range(4) for j in range(4)])
        expected=pair_recurrence_tensor(B,left,4,empty)[0].reshape(4,4,4)*np.exp(c)
        np.testing.assert_allclose(finite,expected,atol=3e-14,rtol=0)

    def test_log_branch_not_principal_determinant_square_root(self):
        n=24; h=.9; d=.9*np.exp(1j*math.pi/3)
        _,c,info=contract_gaussians(h*np.eye(n),d*np.eye(n),n)
        expected=(1-h*d)**(-n//2)
        self.assertAlmostEqual(abs(np.exp(c)/expected-1),0,delta=2e-14)
        self.assertGreater(abs(c.imag),math.pi)
        self.assertLess(info['solve_residual'],1e-14)

    def test_guards_and_no_conjugation(self):
        with self.assertRaises(GaussianContractionError): contract_gaussians(np.eye(1),np.zeros((1,1)),1)
        with self.assertRaises(GaussianContractionError): fold_kernel_chain([np.array([[1.1]])],[0,0])
        with self.assertRaises(ValueError): fold_kernel_chain([np.eye(2)],[0,0])
        with self.assertRaises(ValueError): contract_gaussians(np.eye(1),np.eye(1),True)
        Q=np.array([[0,.2j],[.2j,0]])
        B,_,_=fold_kernel_chain([Q,Q],[0,1,0]); self.assertLess(B[0,1].real,0)


if __name__=='__main__': unittest.main()
