"""Pure-phase conditional rates, boundary selection and complete replay."""
import ctypes as C,os,shutil,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration89_common_v1 import puretree,pnr,tree,small_input,NativeDirectClickSampler,DP,IP,ptr
class NativePureTree89Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_PURETREE89_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build puretree89 probe first')
  cls.dll=Path(path);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,q):
  model=NativeDirectClickSampler(q['v'],q['x'],self.dll,runtime_dirs=self.dirs);self.addCleanup(model.close);handle,destroy=tree(model,4);self.addCleanup(destroy,handle);return model,handle
 def test_photon_subtracted_pair_analytic_rates(self):
  q=small_input(dict(name='paired_strong',squeezing=.85,eta=1.));model,handle=self.make(q);fn=model.lib.pnr89_pure_probe;fn.argtypes=[C.c_void_p,C.c_void_p,DP,DP,C.c_int,IP,C.c_int,DP,IP,DP,DP,C.c_char_p,C.c_int];modes=np.array([0],np.int32);draws=np.array([0.,.1,.49,.5,.9,np.nextafter(1.,0.)]);selected=np.zeros((6,2),np.int32);rates=np.zeros(2);stats=np.zeros(4);err=C.create_string_buffer(4096);rc=fn(model.handle,handle,ptr(q['t']),ptr(q['mean']),1,modes.ctypes.data_as(IP),6,ptr(draws),selected.ctypes.data_as(IP),ptr(rates),ptr(stats),err,len(err));self.assertEqual(rc,0,err.value);n=.85**2/(1-.85**2);np.testing.assert_allclose(rates,np.array([2*n,2*n+1])/(4*n+1),rtol=2e-13,atol=2e-14);np.testing.assert_array_equal(selected[:,0],selected[:,1])
 def test_original_evolution_and_rng_stats_preserved(self):
  rng=np.random.default_rng(890204);v=np.linalg.qr(rng.normal(size=(17,3))+1j*rng.normal(size=(17,3)))[0];t=np.array([.15,.35,.65]);q=dict(v=v,t=t,x=np.diag(np.r_[-t/(1+t),t/(1-t)]),L=np.eye(6)*.1,mean=np.zeros(6));model,handle=self.make(q)
  for R in (3,5):
   old,om=pnr(model,handle,q,R,64,0x89000101);new,nm=puretree(model,handle,q,R,64,0x89000101);self.assertEqual(om['returncode'],0,om);self.assertEqual(nm['returncode'],0,nm)
   for key in ('counts','errors','valid'):np.testing.assert_array_equal(old[key],new[key])
   np.testing.assert_array_equal(old['statistics'][:,:16],new['statistics'][:,:16])
 def test_threads_chunks_include_failed_rows(self):
  q=small_input(dict(name='paired_strong',squeezing=.85,eta=1.));model,handle=self.make(q);whole,_=puretree(model,handle,q,3,256,0x88001013,threads=4);left,_=puretree(model,handle,q,3,61,0x88001013,threads=1);right,_=puretree(model,handle,q,3,195,0x88001013,offset=61,threads=2)
  for key in whole:np.testing.assert_array_equal(whole[key],np.concatenate([left[key],right[key]]))
