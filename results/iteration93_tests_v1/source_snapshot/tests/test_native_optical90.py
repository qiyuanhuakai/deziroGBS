"""Actual output-space constraints, cache ownership and reproducible failures."""
import ctypes as C,os,shutil,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration90_common_v1 import optical_cache,optical_sample,uniforms,NativeStableClickSampler,ptr,DP
class Optical90Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_OPTICAL90_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build optical90 explicitly')
  cls.dll=Path(path);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,v,x):
  model=NativeStableClickSampler(v,x,self.dll,branch_budget=128,runtime_dirs=self.dirs);self.addCleanup(model.close);cache,destroy,meta=optical_cache(model);self.addCleanup(destroy,cache);return model,cache,meta
 def test_actual_map_accepts_valid_remaining_state(self):
  model,cache,meta=self.make(np.eye(1,dtype=complex),np.eye(2));fn=model.lib.tc90_optical_probe;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];q=-.6/1.6+.01;p=.6/.4+.01;x=np.diag([q/(1+.5*q),p/(1+.5*p)]);v=np.array([[np.sqrt(.5)+0j]]);stats=np.zeros(6);err=C.create_string_buffer(4096);rc=fn(1,1,ptr(x),ptr(v.view(float)),ptr(stats),err,len(err));self.assertEqual(rc,0,err.value);self.assertEqual(stats[0],1);self.assertEqual(stats[5],0);self.assertLess(stats[2],stats[1]/4)
 def test_cache_cannot_be_used_for_different_optical_map(self):
  first,cache,meta=self.make(np.eye(2,dtype=complex),np.eye(4));second,_,_=self.make(np.array([[0,1],[1,0]],complex),np.eye(4));raw,result=optical_sample(second,cache,np.full((2,2),.5),2);self.assertNotEqual(result['returncode'],0);self.assertIn('model mismatch',result['error']);self.assertLess(meta['gram_residual'],1e-15)
 def test_frozen_failed_address_finishes(self):
  z=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');model,cache,meta=self.make(z['v'],z['x'][1]);raw,result=optical_sample(model,cache,uniforms(0x88004002,1,685,1,len(z['v'])),16,threads=1);self.assertEqual(result['returncode'],0,result);self.assertEqual(raw['statistics'][0,5],1152);self.assertTrue(np.all((raw['conditional']>=0)&(raw['conditional']<=1)))
 def test_thread_chunk_replay_keeps_explicit_bad_draw(self):
  rng=np.random.default_rng(900101);v=np.linalg.qr(rng.normal(size=(12,3))+1j*rng.normal(size=(12,3)))[0];t=np.array([.2,.4,.6]);model,cache,_=self.make(v,.8*np.diag(np.r_[-t/(1+t),t/(1-t)]));draws=rng.random((13,12));draws[3,5]=np.nan;whole,meta=optical_sample(model,cache,draws,2,threads=4);left,_=optical_sample(model,cache,draws[:6],2,threads=1);right,_=optical_sample(model,cache,draws[6:],2,threads=2);self.assertNotEqual(meta['returncode'],0);self.assertEqual(whole['errors'][3,1],5)
  for k in whole:np.testing.assert_array_equal(whole[k],np.concatenate([left[k],right[k]]))
