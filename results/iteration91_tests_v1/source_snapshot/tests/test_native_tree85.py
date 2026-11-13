"""Optical-tree PNR detector identities, failure guards and domain replay."""
import ctypes as C
import functools
import itertools
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_direct_click import NativeDirectClickSampler
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
class NativeTree85Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  value=os.environ.get('GBSLAB_TREE85_LIBRARY')
  if not value or not Path(value).is_file():raise unittest.SkipTest('Build the explicit iteration85 tree library first')
  cls.library=Path(value);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,coherent=False):
  rng=np.random.default_rng(850123);v=np.linalg.qr(rng.normal(size=(5,2))+1j*rng.normal(size=(5,2)))[0];t=np.array([.25,.45]);x=np.diag(np.r_[-t/(1+t),t/(1-t)]);model=NativeDirectClickSampler(v,x,self.library,runtime_dirs=self.dirs);lib=model.lib
  lib.pnr85_tree_create.argtypes=[C.c_void_p,C.c_char_p,C.c_int];lib.pnr85_tree_create.restype=C.c_void_p;lib.pnr85_tree_destroy.argtypes=[C.c_void_p];err=C.create_string_buffer(4096);tree=lib.pnr85_tree_create(model.handle,err,len(err));self.assertTrue(tree,err.value)
  self.addCleanup(model.close);self.addCleanup(lib.pnr85_tree_destroy,tree);return model,tree,v,t,x
 def probe(self,model,tree,x,mean,modes,draws):
  x=np.ascontiguousarray(x);mean=np.ascontiguousarray(mean);modes=np.ascontiguousarray(modes,np.int32);draws=np.ascontiguousarray(draws,float);old=np.zeros(model.modes);new=old.copy();selected=np.zeros((len(draws),2),np.int32);stats=np.zeros(2);err=C.create_string_buffer(4096);fn=model.lib.pnr85_detector_probe;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,IP,C.c_int,DP,DP,DP,IP,DP,C.c_char_p,C.c_int];rc=fn(model.handle,tree,ptr(x),ptr(mean),len(modes),modes.ctypes.data_as(IP),len(draws),ptr(draws),ptr(old),ptr(new),selected.ctypes.data_as(IP),ptr(stats),err,len(err));return rc,err.value,old,new,selected,stats
 def test_conditional_rates_match_independent_real_wick(self):
  model,tree,v,t,x=self.make();x=x+.3*np.eye(4);mean=np.array([.7,-.3,.4,.6]);a=np.r_[v.real.T,-v.imag.T];b=np.r_[v.imag.T,v.real.T]
  for modes in ([],[1],[1,1],[0,2,4]):
   rates=[]
   for mode in range(5):
    fs=list(modes)+[mode];value=0.
    for choices in itertools.product((0,1),repeat=len(fs)):
     F=np.column_stack([(a if j==0 else b)[:,i] for i,j in zip(fs,choices)]);mu=F.T@mean;cov=F.T@x@F
     @functools.lru_cache(None)
     def moment(ids):
      if not ids:return 1.
      first,*rest=ids;value=mu[first]*moment(tuple(rest))
      for k,j in enumerate(rest):value+=cov[first,j]*moment(tuple(rest[:k]+rest[k+1:]))
      return value
     value+=moment(tuple(j for i in range(len(fs)) for j in (i,i)))
    rates.append(value/2**len(fs))
   target=np.array(rates)/sum(rates);rc,error,old,new,selected,stats=self.probe(model,tree,x,mean,modes,np.linspace(0,.999,21));self.assertEqual(rc,0,error);np.testing.assert_allclose(old,target,rtol=2e-12,atol=2e-14);np.testing.assert_allclose(new,target,rtol=2e-12,atol=2e-14);np.testing.assert_array_equal(selected[:,0],selected[:,1])
 def test_boundaries_fall_back_without_changing_selection(self):
  model,tree,v,t,x=self.make();mean=np.array([.7,-.3,.4,.6]);x=x+.3*np.eye(4);r=self.probe(model,tree,x,mean,[1,1],[.3]);self.assertEqual(r[0],0,r[1]);cdf=np.cumsum(r[2])[:-1];draws=np.r_[0,np.nextafter(1.,0.),cdf,np.nextafter(cdf,0),np.nextafter(cdf,1)];r=self.probe(model,tree,x,mean,[1,1],draws);self.assertEqual(r[0],0,r[1]);np.testing.assert_array_equal(r[4][:,0],r[4][:,1]);self.assertGreater(r[5][1],0)
 def test_coherent_rank_deficiency_preserves_old_rates(self):
  model,tree,v,t,x=self.make();mean=np.array([1.,0.,0.,0.]);r=self.probe(model,tree,np.zeros((4,4)),mean,[],np.linspace(0,.999,11));self.assertEqual(r[0],0,r[1]);target=abs(v[:,0])**2;np.testing.assert_allclose(r[3],target,atol=2e-15);np.testing.assert_array_equal(r[4][:,0],r[4][:,1]);self.assertEqual(r[5][1],len(r[4]))
 def test_tree_rejects_different_optics(self):
  model,tree,v,t,x=self.make();other=NativeDirectClickSampler(v[::-1].copy(),x,self.library,runtime_dirs=self.dirs)
  with other:r=self.probe(other,tree,x,np.zeros(4),[],[.5]);self.assertNotEqual(r[0],0);self.assertIn(b'mismatch',r[1])
 def sample(self,model,tree,t,count,offset,threads,old=False,resplit=0):
  L=np.eye(4)*.4;mean=np.zeros(4);out=np.zeros((count,5),np.uint32);stats=np.zeros((count,16 if old else 18));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096);common=[DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];args=[ptr(t),ptr(L),count,0x8500a001,2,offset,threads,1,resplit,ptr(mean),None,out.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err)]
  fn=getattr(model.lib,'pnr84d_correlated_sample' if old else 'pnr85_correlated_sample');fn.argtypes=[C.c_void_p,*common] if old else [C.c_void_p,C.c_void_p,*common];rc=fn(model.handle,*args) if old else fn(model.handle,tree,*args);return rc,err.value,out,stats,errors
 def test_complete_replay_and_batch_thread_invariance(self):
  model,tree,v,t,x=self.make();old=self.sample(model,tree,t,11,0,1,True);new=self.sample(model,tree,t,11,0,4);left=self.sample(model,tree,t,3,0,2);right=self.sample(model,tree,t,8,3,2)
  for r in (old,new,left,right):self.assertEqual(r[0],0,r[1]);self.assertFalse(r[4].any())
  np.testing.assert_array_equal(old[2],new[2]);np.testing.assert_array_equal(old[3],new[3][:,:16]);np.testing.assert_array_equal(new[2],np.concatenate([left[2],right[2]]));np.testing.assert_array_equal(new[3],np.concatenate([left[3],right[3]]))
 def test_unsupported_resplitting_is_rejected(self):
  model,tree,v,t,x=self.make();r=self.sample(model,tree,t,1,0,1,resplit=1);self.assertNotEqual(r[0],0);self.assertIn(b'Invalid',r[1])
