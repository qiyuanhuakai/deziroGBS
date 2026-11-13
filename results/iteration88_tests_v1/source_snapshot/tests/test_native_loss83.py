import os
from pathlib import Path
import unittest
import numpy as np
from gbslab.native_loss_map import prepare_lossy_gaussian_map

class NativeLoss83Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('GBSLAB_LOSS83_LIBRARY')
        if not path or not Path(path).is_file():raise unittest.SkipTest('Build the native loss83 library explicitly')
        cls.library=path
        cls.runtime=[p for p in os.environ.get('GBSLAB_NATIVE_RUNTIME_DIRS','').split(os.pathsep) if p]
    def call(self,t,x):return prepare_lossy_gaussian_map(t,x,self.library,runtime_dirs=self.runtime)
    @staticmethod
    def realify(t):return np.block([[t.real,-t.imag],[t.imag,t.real]])
    def test_random_complex_loss_reconstructs_all_covariance_entries(self):
        rng=np.random.default_rng(830001);q,_=np.linalg.qr(rng.normal(size=(12,3))+1j*rng.normal(size=(12,3)));r=np.array([[.3,.04j,.02],[0,.5,.01j],[0,0,.7]]);t=q@r;x=np.diag(np.r_[np.expm1(-2*np.array([.2,.4,.6])),np.expm1(2*np.array([.2,.4,.6]))])/2
        got=self.call(t,x);v,y=got['isometry'],got['excess_covariances'][0];np.testing.assert_allclose(v.conj().T@v,np.eye(3),atol=1e-14);np.testing.assert_allclose(self.realify(v)@y@self.realify(v).T,self.realify(t)@x@self.realify(t).T,atol=2e-14)
    def test_sectors_kept_separate(self):
        t=np.eye(3)*.6;xs=np.array([np.eye(6)*.1,np.eye(6)*.7]);out=self.call(t,xs)['excess_covariances'];np.testing.assert_allclose(out,xs*.36,atol=1e-14)
    def test_zero_and_rank_deficient_transmission(self):
        for t in (np.zeros((4,2)),np.array([[.5,0],[0,0],[0,0],[0,0]])):
            out=self.call(t,np.eye(4));v,y=out['isometry'],out['excess_covariances'][0];np.testing.assert_allclose(self.realify(v)@y@self.realify(v).T,self.realify(t)@self.realify(t).T,atol=1e-14)
    def test_vacuum_is_preserved(self):
        out=self.call(np.eye(2)*.4,np.zeros((4,4)));self.assertFalse(out['excess_covariances'].any())
    def test_active_map_rejected(self):
        with self.assertRaisesRegex(ValueError,'passive contraction'):self.call(np.eye(2)*1.1,np.eye(4))
    def test_nonphysical_covariance_rejected(self):
        with self.assertRaisesRegex(ValueError,'physicality'):self.call(np.eye(2)*.5,-np.eye(4))
    def test_nonsymmetric_covariance_rejected(self):
        x=np.eye(4);x[0,1]=.1
        with self.assertRaisesRegex(ValueError,'symmetric'):self.call(np.eye(2)*.5,x)
    def test_bad_shape_and_nonfinite_inputs(self):
        for t,x in ((np.ones((1,2)),np.eye(4)),(np.eye(2),np.eye(3)),(np.eye(2),np.eye(4)*1j),(np.eye(2)*np.nan,np.eye(4))):
            with self.assertRaises(ValueError):self.call(t,x)
