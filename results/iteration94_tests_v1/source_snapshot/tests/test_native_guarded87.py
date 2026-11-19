"""Analytic weak thermal conditionals, correlated clicks and guarded boundaries."""
import ctypes as C,os,shutil,unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration87_guarded_common_v1 import sample,compare,NativeStableClickSampler
class NativeGuarded87Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  value=os.environ.get('GBSLAB_GUARDED87_LIBRARY')
  if not value or not Path(value).is_file():raise unittest.SkipTest('Build iteration87 guarded click first')
  cls.dll=Path(value);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,v,x):
  model=NativeStableClickSampler(v,x,self.dll,branch_budget=128,runtime_dirs=self.dirs);self.addCleanup(model.close);return model
 def test_weak_independent_conditionals_all_budgets(self):
  n=np.array([.001,.002,.003,.004,.005,.006,.007,.008]);model=self.make(np.eye(8,dtype=complex),np.diag(np.r_[n,n]));draws=np.array([[0. if label>>j&1 else np.nextafter(1.,0.) for j in range(8)] for label in range(256)])
  for budget in (1,8,64,128):
   raw,meta=sample(model,draws,budget);self.assertEqual(meta['returncode'],0,meta);target=np.tile(n/(1+n),(256,1));ball=raw['kind']==1;np.testing.assert_allclose(raw['conditional'][~ball],target[~ball],atol=2e-16,rtol=1e-13);self.assertTrue(np.all(raw['bounds'][:,:,0][ball]<=target[ball]+2e-16));self.assertTrue(np.all(target[ball]<=raw['bounds'][:,:,1][ball]+2e-16));old,om=sample(model,draws,budget,False);self.assertEqual(om['returncode'],0,om);c=compare(old,raw);self.assertTrue(c['equal_decisions_errors_state_counters']);self.assertTrue(c['oracle_probabilities_enclosed']);self.assertTrue(c['counter_identities'])
 def test_squeezed_pair_identical_clicks_and_vacuum(self):
  x=np.array([[9,15,0,0],[15,9,0,0],[0,0,9,-15],[0,0,-15,9]],float)/16;model=self.make(np.eye(2,dtype=complex),x);draws=np.array([[0.,.9],[.35,.8],[.36,0.],[.9,0.]]);raw,meta=sample(model,draws,64);self.assertEqual(meta['returncode'],0,meta);np.testing.assert_array_equal(raw['clicks'][:,0],raw['clicks'][:,1]);vac=self.make(np.eye(3,dtype=complex),np.zeros((6,6)));raw,meta=sample(vac,np.zeros((4,3)),64);self.assertEqual(meta['returncode'],0,meta);self.assertFalse(raw['clicks'].any());self.assertFalse(raw['conditional'].any())
 def test_thread_chunks_preserve_intervals_and_counters(self):
  rng=np.random.default_rng(870112);v=np.linalg.qr(rng.normal(size=(9,3))+1j*rng.normal(size=(9,3)))[0];t=np.array([.2,.3,.5]);model=self.make(v,.6*np.diag(np.r_[-t/(1+t),t/(1-t)]));draws=rng.random((11,9));whole,wm=sample(model,draws,8,threads=4);left,lm=sample(model,draws[:3],8,threads=1);right,rm=sample(model,draws[3:],8,threads=2)
  for meta in (wm,lm,rm):self.assertEqual(meta['returncode'],0,meta)
  for k in whole:np.testing.assert_array_equal(whole[k],np.concatenate([left[k],right[k]]))
 def test_invalid_uniform_preserves_failure_row(self):
  model=self.make(np.eye(2,dtype=complex),np.eye(4)*.2);raw,meta=sample(model,np.array([[.5,np.nan],[.2,.3]]),64);self.assertNotEqual(meta['returncode'],0);self.assertEqual(raw['errors'][0,1],1);self.assertEqual(raw['errors'][1,0],0);self.assertEqual(raw['statistics'][1,5],2);self.assertEqual(raw['kind'][0,1],255)
