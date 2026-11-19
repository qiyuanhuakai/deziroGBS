"""Analytic independent modes, conditioned correlations and explicit RNG guards."""
import ctypes as C
import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_stable_click import NativeStableClickSampler
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int)
def ptr(x):return x.ctypes.data_as(DP)

class NativeShared86Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  value=os.environ.get('GBSLAB_SHARED86_LIBRARY')
  if not value or not Path(value).is_file():raise unittest.SkipTest('Build iteration86 shared Gaussian library first')
  cls.library=Path(value);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,v,x):
  model=NativeStableClickSampler(v,x,self.library,branch_budget=64,runtime_dirs=self.dirs);self.addCleanup(model.close);model.lib.tc86_shared_prefix.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];model.lib.tc86_shared_sample.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];model.lib.tc86_shared_conditionals.argtypes=[C.c_void_p,C.c_int,BP,C.c_int,DP,DP,C.c_char_p,C.c_int];return model
 def prefix(self,model,bits,budget=64):
  bits=np.ascontiguousarray(bits,np.uint8);p=np.zeros(1);s=np.zeros(6);e=C.create_string_buffer(4096);rc=model.lib.tc86_shared_prefix(model.handle,len(bits),bits.ctypes.data_as(BP),budget,ptr(p),ptr(s),e,len(e));return rc,e.value,p[0],s
 def sample(self,model,draws,threads=1,budget=64,missing=False):
  draws=np.ascontiguousarray(draws,float);b=np.zeros(draws.shape,np.uint8);p=np.zeros(draws.shape);e=np.zeros((len(draws),2),np.int32);s=np.zeros((len(draws),6));message=C.create_string_buffer(4096);rc=model.lib.tc86_shared_sample(model.handle,len(draws),threads,budget,None if missing else ptr(draws),b.ctypes.data_as(BP),ptr(p),e.ctypes.data_as(IP),ptr(s),message,len(message));return rc,message.value,b,p,e,s
 def test_independent_thermal_complete_law_all_budgets(self):
  n=np.array([.125,.25,.5,1.]);model=self.make(np.eye(4,dtype=complex),np.diag(np.r_[n,n]));click=n/(1+n)
  for budget in (1,2,8,64,128):
   total=0.
   for label in range(16):
    bits=np.array([(label>>j)&1 for j in range(4)]);r=self.prefix(model,bits,budget);self.assertEqual(r[0],0,r[1]);self.assertAlmostEqual(r[2],np.prod(np.where(bits,click,1-click)),places=14);total+=r[2]
   self.assertAlmostEqual(total,1.,places=14)
 def test_squeezed_pair_has_identical_click_bits_without_closure(self):
  x=np.array([[9,15,0,0],[15,9,0,0],[0,0,9,-15],[0,0,-15,9]],float)/16;model=self.make(np.eye(2,dtype=complex),x)
  for bits,expected in [([0,0],16/25),([1,1],9/25),([0,1],0),([1,0],0)]:
   r=self.prefix(model,bits);self.assertEqual(r[0],0,r[1]);self.assertAlmostEqual(r[2],expected,places=14)
 def test_vacuum_never_clicks_even_at_zero_uniform(self):
  model=self.make(np.eye(3,dtype=complex),np.zeros((6,6)));r=self.sample(model,np.zeros((4,3)),threads=3);self.assertEqual(r[0],0,r[1]);self.assertFalse(r[2].any());self.assertFalse(r[3].any());self.assertTrue((r[5][:,5]==3).all())
 def test_forced_prefix_matches_sample_and_thread_chunks(self):
  rng=np.random.default_rng(860123);v=np.linalg.qr(rng.normal(size=(7,3))+1j*rng.normal(size=(7,3)))[0];t=np.array([.2,.4,.6]);x=.7*np.diag(np.r_[-t/(1+t),t/(1-t)]);model=self.make(v,x);draws=rng.random((9,7));whole=self.sample(model,draws,4,2);left=self.sample(model,draws[:3],1,2);right=self.sample(model,draws[3:],2,2)
  for r in (whole,left,right):self.assertEqual(r[0],0,r[1])
  for j in range(2,6):np.testing.assert_array_equal(whole[j],np.concatenate([left[j],right[j]]))
  for row in range(9):
   bits=np.ascontiguousarray(whole[2][row,:6]);p=np.zeros(7);s=np.zeros(6);error=C.create_string_buffer(4096);rc=model.lib.tc86_shared_conditionals(model.handle,6,bits.ctypes.data_as(BP),2,ptr(p),ptr(s),error,len(error));self.assertEqual(rc,0,error.value);np.testing.assert_array_equal(p,whole[3][row])
 def test_explicit_random_input_and_illegal_bits_rejected(self):
  model=self.make(np.eye(2,dtype=complex),np.eye(4)*.5);missing=self.sample(model,np.zeros((1,2)),missing=True);self.assertNotEqual(missing[0],0);self.assertIn(b'explicit draws',missing[1]);bad=self.sample(model,[[.5,float('nan')]]);self.assertNotEqual(bad[0],0);self.assertEqual(bad[4][0,1],1);self.assertNotEqual(self.prefix(model,[2,0])[0],0)
 def test_sse_rounding_is_restored_and_does_not_change_probability(self):
  model=self.make(np.eye(2,dtype=complex),np.eye(4)*.5);fn=model.lib.tc86_rounding_test;fn.argtypes=[C.c_void_p,C.c_int,BP,DP,C.POINTER(C.c_uint),C.c_char_p,C.c_int];bits=np.array([1,0],np.uint8);values=[]
  for mode in range(4):
   value=np.zeros(1);state=(C.c_uint*2)();error=C.create_string_buffer(4096);rc=fn(model.handle,mode,bits.ctypes.data_as(BP),ptr(value),state,error,len(error));self.assertEqual(rc,0,error.value);self.assertEqual(state[0]&0x6000,mode<<13);self.assertEqual(state[1]&0x6000,mode<<13);values.append(value[0])
  self.assertTrue(all(p==values[0] for p in values));self.assertAlmostEqual(values[0],2/9,places=14)

if __name__=='__main__':unittest.main()
