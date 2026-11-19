"""General squeezing, displacement, complex interference, trace budgets and domains."""
import os,sys,ctypes as C,shutil,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration93_common_v1 import local93,sample93
from iteration93_identity_v2 import local_map
class Mps93Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_MPS93_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build Gaussian MPS93 explicitly')
  cls.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));cls.lib=C.CDLL(path)
 def test_displaced_local_map(self):
  for t in (0.,.4,-.6):
   for eta in (0.,.17,.8,1.):
    mat,s=local93(self.lib,6,t,.2+.3j,eta);M,f,tp,ap=local_map(6,t,.2+.3j,eta);np.testing.assert_allclose(mat,M,rtol=1e-12,atol=3e-14);np.testing.assert_allclose(s,[tp,ap.real,ap.imag,2*np.log(abs(f))],atol=3e-14)
 def test_vacuum_many_modes(self):
  raw,meta=sample93(self.lib,np.eye(5),np.zeros(5),np.zeros(5),32,0x93000100);self.assertEqual(meta['returncode'],0,meta);self.assertFalse(raw['counts'].any());np.testing.assert_array_equal(raw['statistics'][:,14],1)
 def test_squeezed_parity(self):
  raw,meta=sample93(self.lib,np.eye(1),[.3],[0j],256,0x93000101,d=16,bond=16,budget=.001);self.assertEqual(meta['returncode'],0,meta);self.assertFalse((raw['counts']%2).any())
 def test_coherent_dark_port(self):
  u=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2);raw,meta=sample93(self.lib,u,[0,0],[.5,.5j],256,0x93000102);self.assertEqual(meta['returncode'],0,meta);self.assertFalse(raw['counts'][:,0].any());self.assertTrue(raw['counts'][:,1].any())
 def test_paired_charge(self):
  u=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);raw,meta=sample93(self.lib,u,[.5,-.5],[0,0],256,0x93000103,d=16,bond=32,budget=.001);self.assertEqual(meta['returncode'],0,meta);np.testing.assert_array_equal(raw['counts'][:,0],raw['counts'][:,1])
 def test_bond_budget_failures_are_retained(self):
  u=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);raw,meta=sample93(self.lib,u,[.6,-.6],[0,0],128,0x93000104,d=8,bond=1,budget=0);self.assertEqual(meta['returncode'],1,meta);self.assertTrue((raw['errors']==2).any());self.assertFalse((raw['errors']==4).any());self.assertFalse(raw['counts'].any());self.assertFalse(raw['statistics'][:,5].any())
 def test_worker_and_batch_replay_with_displacement_noise(self):
  rng=np.random.default_rng(93);u=np.linalg.qr(rng.normal(size=(4,3))+1j*rng.normal(size=(4,3)))[0];kwargs=dict(d=5,bond=8,budget=.1,noise=np.eye(6)*.05);a,meta=sample93(self.lib,u,[.2,-.3,.4],[.1j,.2,-.1],64,0x93000105,sector=2,offset=7,threads=1,**kwargs);self.assertEqual(meta['returncode'],0,meta);parts=[sample93(self.lib,u,[.2,-.3,.4],[.1j,.2,-.1],n,0x93000105,sector=2,offset=7+off,threads=4,**kwargs)[0] for off,n in ((0,23),(23,41))]
  for key in a:np.testing.assert_array_equal(a[key],np.concatenate([p[key] for p in parts]))
 def test_overflow_and_isometry_rejected(self):
  for u,sector,offset in ((np.ones((2,2)),0,0),(np.eye(2),65536,0),(np.eye(2),0,2**32)):
   _,meta=sample93(self.lib,u,[.2,.3],[0,0],2,0x93000106,sector=sector,offset=offset);self.assertEqual(meta['returncode'],2)
