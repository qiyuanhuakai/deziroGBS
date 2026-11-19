"""Analytic and domain gates for the fixed-basis convex pure-core optimizer."""
import ctypes as C
import os
from pathlib import Path
import unittest
import numpy as np

class NativeCore83Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('GBSLAB_CORE83_LIBRARY')
        if not path or not Path(path).is_file():raise unittest.SkipTest('Build core83 explicitly')
        cls.runtime=[os.add_dll_directory(p) for p in os.environ.get('GBSLAB_NATIVE_RUNTIME_DIRS','').split(os.pathsep) if p]
        cls.lib=C.CDLL(path);cls.dp=C.POINTER(C.c_double);cls.fn=cls.lib.core83_optimize;cls.fn.argtypes=[C.c_int,cls.dp,cls.dp,cls.dp,cls.dp,cls.dp,cls.dp,cls.dp,C.c_char_p,C.c_int];cls.fn.restype=C.c_int
    @classmethod
    def tearDownClass(cls):
        for h in cls.runtime:h.close()
    def call(self,x,R,t):
        x=np.ascontiguousarray(x,float);R=np.ascontiguousarray(R,float);t=np.ascontiguousarray(t,float);out=np.zeros_like(t);noise=np.zeros_like(x);dual=np.zeros_like(x);stats=np.zeros(8);err=C.create_string_buffer(4096);ptr=lambda x:x.ctypes.data_as(self.dp);rc=self.fn(len(t),ptr(x),ptr(R),ptr(t),ptr(out),ptr(noise),ptr(dual),ptr(stats),err,len(err))
        if rc:raise ValueError(err.value.decode())
        return out,noise,dual,stats
    def test_one_mode_matches_closed_form_minimum_core(self):
        a,b=.6,3.;q=np.sqrt(a/b);t=np.array([(1-q)/(1+q)]);out,noise,dual,stats=self.call((np.diag([a,b])-np.eye(2))/2,np.eye(2),t);exact=(a+1/a-2)/4;self.assertLess(abs(stats[1]-exact),1e-6);self.assertLessEqual(stats[2],exact+1e-7);self.assertGreater(np.linalg.eigvalsh(noise).min(),0);self.assertLess(stats[3],1e-5)
    def test_classical_thermal_state_has_vacuum_core(self):
        out,noise,dual,stats=self.call(np.eye(4)*.5,np.eye(4),np.zeros(2));np.testing.assert_allclose(out,0,atol=1e-12);np.testing.assert_allclose(noise,np.eye(4)*.5,atol=1e-12);self.assertAlmostEqual(stats[1],0,places=12)
    def test_rotated_two_mode_closed_form_and_reconstruction(self):
        U=np.array([[1,1j],[1j,1]])/np.sqrt(2);R=np.block([[U.real,-U.imag],[U.imag,U.real]]);a=np.array([.4,.7]);b=np.array([4.,3.]);V=(R*np.r_[a,b])@R.T;q=np.sqrt(a/b);t=(1-q)/(1+q);out,noise,dual,stats=self.call((V-np.eye(4))/2,R,t);pure=(R*np.r_[(1-out)/(1+out),(1+out)/(1-out)])@R.T;np.testing.assert_allclose(pure+2*noise,V,atol=1e-12);self.assertLess(abs(stats[1]-np.sum(a+1/a-2)/4),1e-6)
    def test_nonpassive_basis_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'passive'):self.call(np.eye(2),np.diag([2.,.5]),np.array([0.]))
    def test_infeasible_initial_core_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'positive-noise'):self.call(np.eye(2)*.1,np.eye(2),np.array([.9]))
    def test_invalid_squeezing_and_nonsymmetric_input_rejected(self):
        with self.assertRaisesRegex(ValueError,'squeezing'):self.call(np.eye(2),np.eye(2),np.array([1.]))
        with self.assertRaisesRegex(ValueError,'covariance'):self.call(np.array([[1.,.2],[0,1.]]),np.eye(2),np.array([0.]))
