"""Physical interior/rejection, analytic thermal future and frozen failure replay."""
import ctypes as C,os,shutil,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration89_common_v1 import physical,uniforms,NativeStableClickSampler,DP,ptr
class NativePhysical89Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_PHYSICAL89_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build physical89 first')
  cls.dll=Path(path);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,v,x):
  model=NativeStableClickSampler(v,x,self.dll,branch_budget=128,runtime_dirs=self.dirs);self.addCleanup(model.close);return model
 def test_uncertainty_interior_boundary_and_negative(self):
  model=self.make(np.eye(2,dtype=complex),np.eye(4));fn=model.lib.tc89_physical_probe;fn.argtypes=[C.c_int,DP,DP,C.c_char_p,C.c_int]
  for x,expected in ((np.eye(4)*.1,1),(np.zeros((4,4)),0),(np.diag([-.2,.1,.1,.1]),0)):
   stats=np.zeros(4);err=C.create_string_buffer(4096);rc=fn(4,ptr(x),ptr(stats),err,len(err));self.assertEqual(rc,0,err.value);self.assertEqual(stats[0],expected)
   if expected:self.assertLess(stats[2],stats[1]/4)
 def test_old_failed_record_finishes_without_clipping(self):
  z=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');model=self.make(z['v'],z['x'][1]);raw,meta=physical(model,uniforms(0x88004002,1,685,1,len(z['v'])),16,threads=1);self.assertEqual(meta['returncode'],0,meta);self.assertEqual(raw['statistics'][0,5],1152);self.assertGreater(raw['statistics'][0,20],0);self.assertTrue(np.all((raw['conditional']>=0)&(raw['conditional']<=1)))
 def test_thermal_exact_future_and_rejection_accounting(self):
  n=np.linspace(.1,.8,8);model=self.make(np.eye(8,dtype=complex),np.diag(np.r_[n,n]));draws=np.array([[0. if i>>j&1 else np.nextafter(1.,0.) for j in range(8)] for i in range(256)])
  for budget in (1,2,8,64):
   raw,meta=physical(model,draws,budget);self.assertEqual(meta['returncode'],0,meta);np.testing.assert_allclose(raw['conditional'],np.tile(n/(1+n),(256,1)),atol=2e-14,rtol=2e-13);s=raw['statistics'];np.testing.assert_array_equal(s[:,12],s[:,13]+s[:,14]);np.testing.assert_array_equal(s[:,14],s[:,16:21].sum(1))
 def test_threads_chunks_and_explicit_failure(self):
  rng=np.random.default_rng(890105);v=np.linalg.qr(rng.normal(size=(12,3))+1j*rng.normal(size=(12,3)))[0];t=np.array([.2,.4,.6]);model=self.make(v,.8*np.diag(np.r_[-t/(1+t),t/(1-t)]));draws=rng.random((13,12));draws[3,5]=np.nan;whole,meta=physical(model,draws,2,threads=4);left,_=physical(model,draws[:6],2,threads=1);right,_=physical(model,draws[6:],2,threads=2);self.assertNotEqual(meta['returncode'],0);self.assertEqual(whole['errors'][3,1],5)
  for key in whole:np.testing.assert_array_equal(whole[key],np.concatenate([left[key],right[key]]))
