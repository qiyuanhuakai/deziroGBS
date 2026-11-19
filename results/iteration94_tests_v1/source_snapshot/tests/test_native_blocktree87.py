"""Unbalanced tree sums, domain replay, detector boundaries and API guards."""
import ctypes as C,os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_direct_click import NativeDirectClickSampler
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);UP=C.POINTER(C.c_uint32)
def ptr(x):return x.ctypes.data_as(DP)
class NativeBlockTree87Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  name=os.environ.get('GBSLAB_BLOCKTREE87_LIBRARY')
  if not name or not Path(name).is_file():raise unittest.SkipTest('Build iteration87 block tree first')
  cls.dll=Path(name);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,block=7):
  rng=np.random.default_rng(870123);v=np.linalg.qr(rng.normal(size=(17,3))+1j*rng.normal(size=(17,3)))[0];t=np.array([.2,.35,.5]);x=np.diag(np.r_[-t/(1+t),t/(1-t)]);model=NativeDirectClickSampler(v,x,self.dll,runtime_dirs=self.dirs);self.addCleanup(model.close);fn=model.lib.pnr87_tree_create;fn.argtypes=[C.c_void_p,C.c_int,C.c_char_p,C.c_int];fn.restype=C.c_void_p;destroy=model.lib.pnr87_tree_destroy;destroy.argtypes=[C.c_void_p];err=C.create_string_buffer(4096);tree=fn(model.handle,block,err,len(err));self.assertTrue(tree,err.value);self.addCleanup(destroy,tree);return model,tree,v,t,x
 def test_unbalanced_binary_gram_order_all_blocks(self):
  sizes=[]
  for block in (1,2,7,32):
   model,tree,v,t,x=self.make(block);fn=model.lib.pnr87_gram_audit;fn.argtypes=[C.c_void_p,C.c_void_p,DP,C.c_char_p,C.c_int];stats=np.zeros(3);err=C.create_string_buffer(4096);rc=fn(model.handle,tree,ptr(stats),err,len(err));self.assertEqual(rc,0,err.value);self.assertGreater(stats[0],0);self.assertEqual(stats[1],0);self.assertEqual(stats[2],0);info=model.lib.pnr87_tree_info;info.argtypes=[C.c_void_p,DP];s=np.zeros(7);self.assertEqual(info(tree,ptr(s)),0);sizes.append(s[1])
  self.assertTrue(all(a>b for a,b in zip(sizes,sizes[1:])))
 def sample(self,model,tree,t,count,offset,threads,old=False,resplit=0):
  L=np.eye(6)*.4;mean=np.zeros(6);out=np.zeros((count,17),np.uint32);stats=np.zeros((count,16 if old else 20));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096);common=[DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,UP,DP,IP,C.c_char_p,C.c_int];args=[ptr(t),ptr(L),count,0x8700a001,2,offset,threads,2,resplit,ptr(mean),None,out.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),err,len(err)];fn=getattr(model.lib,'pnr84d_correlated_sample' if old else 'pnr87_correlated_sample');fn.argtypes=[C.c_void_p,*common] if old else [C.c_void_p,C.c_void_p,*common];rc=fn(model.handle,*args) if old else fn(model.handle,tree,*args);return rc,err.value,out,stats,errors
 def test_complete_domain_thread_chunk_replay(self):
  model,tree,v,t,x=self.make();old=self.sample(model,tree,t,13,0,1,True);new=self.sample(model,tree,t,13,0,4);left=self.sample(model,tree,t,3,0,2);right=self.sample(model,tree,t,10,3,2)
  for r in (old,new,left,right):self.assertEqual(r[0],0,r[1]);self.assertFalse(r[4].any())
  np.testing.assert_array_equal(old[2],new[2]);np.testing.assert_array_equal(old[3],new[3][:,:16])
  for j in (2,3,4):np.testing.assert_array_equal(new[j],np.concatenate([left[j],right[j]]))
 def test_boundaries_and_rank_deficiency_keep_old_selection(self):
  model,tree,v,t,x=self.make();mean=np.array([1.,0.,0.,0.,0.,0.]);x=np.zeros((6,6));cdf=np.cumsum(abs(v[:,0])**2)[:-1];draws=np.ascontiguousarray(np.r_[0.,np.nextafter(1.,0.),cdf,np.nextafter(cdf,0.),np.nextafter(cdf,1.)]);modes=np.zeros(1,np.int32);selected=np.zeros((len(draws),2),np.int32);stats=np.zeros(4);err=C.create_string_buffer(4096);fn=model.lib.pnr87_detector_probe;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,IP,C.c_int,DP,IP,DP,C.c_char_p,C.c_int];rc=fn(model.handle,tree,ptr(x),ptr(mean),0,modes.ctypes.data_as(IP),len(draws),ptr(draws),selected.ctypes.data_as(IP),ptr(stats),err,len(err));self.assertEqual(rc,0,err.value);np.testing.assert_array_equal(selected[:,0],selected[:,1]);self.assertEqual(stats[1],len(draws))
 def test_invalid_block_and_resplitting_rejected(self):
  model,tree,v,t,x=self.make();err=C.create_string_buffer(4096)
  for block in (0,1025):self.assertFalse(model.lib.pnr87_tree_create(model.handle,block,err,len(err)))
  r=self.sample(model,tree,t,1,0,1,resplit=1);self.assertNotEqual(r[0],0)
