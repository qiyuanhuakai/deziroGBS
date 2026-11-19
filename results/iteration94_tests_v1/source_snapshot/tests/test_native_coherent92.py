"""Physical boundary laws, coherent interference and stream-address replay."""
import os,sys,unittest,ctypes as C,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration92_common_v1 import sample,flow,fock_core
from iteration92_identity_v1 import transform

class Coherent92Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_COHERENT92_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build coherent92 explicitly')
  cls.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));cls.lib=C.CDLL(path)
 def test_coherent_flow_and_endpoints(self):
  rng=np.random.default_rng(92);c=np.zeros((6,6),complex)
  for a in range(6):
   for b in range(6-a):c[a,b]=rng.normal()+1j*rng.normal()
  c/=np.linalg.norm(c)
  for eta in (0.,.001,.31,.9,1.):
   d,p=flow(self.lib,c,.72,eta);ref,_=transform(c,.72,eta);np.testing.assert_allclose(d,ref,atol=3e-15,rtol=3e-14);self.assertAlmostEqual(p,float(np.vdot(ref,ref).real),places=13)
 def test_negative_binomial_survival(self):
  for k in (0,1,4,9):
   for eta in (0.,.1,.7,1.):
    _,p=flow(self.lib,fock_core(k),.85,eta);self.assertAlmostEqual(p,eta**k*((1-.85**2)/(1-.85**2*eta**2))**(k+1),places=13)
 def test_identity_charge_support(self):
  for k in (0,1,4):
   raw,meta=sample(self.lib,np.eye(2),fock_core(k),.7,128,0x92000100+k)
   self.assertEqual(meta['returncode'],0,meta);np.testing.assert_array_equal(raw['counts'][:,0].astype(int)-raw['counts'][:,1],k)
 def test_coherent_dark_port(self):
  c=np.zeros((2,2),complex);c[1,0]=1/np.sqrt(2);c[0,1]=1j/np.sqrt(2);u=np.array([[1,-1j],[1,1j]],complex)/np.sqrt(2)
  raw,meta=sample(self.lib,u,c,0,128,0x92000110);self.assertEqual(meta['returncode'],0,meta);np.testing.assert_array_equal(raw['counts'],np.tile([1,0],(128,1)))
 def test_hong_ou_mandel_even_outputs(self):
  u=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);raw,meta=sample(self.lib,u,fock_core(0),.7,256,0x92000111);self.assertEqual(meta['returncode'],0,meta);self.assertFalse((raw['counts']%2).any())
 def test_thread_and_batch_replay(self):
  u=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2);whole,meta=sample(self.lib,u,fock_core(2),.6,96,0x92000112,sector=3,offset=9,threads=1);self.assertEqual(meta['returncode'],0,meta)
  parts=[sample(self.lib,u,fock_core(2),.6,n,0x92000112,sector=3,offset=9+off,threads=4)[0] for off,n in ((0,31),(31,65))]
  for key in whole:np.testing.assert_array_equal(whole[key],np.concatenate([p[key] for p in parts]))
 def test_failure_is_retained(self):
  raw,meta=sample(self.lib,np.eye(2),fock_core(0),.85,128,0x92000113,cap=0);self.assertEqual(meta['returncode'],1);self.assertTrue((raw['errors']==1).any());self.assertFalse((raw['errors']==2).any());self.assertFalse(raw['counts'].any());np.testing.assert_array_equal(raw['statistics'][:,5],1)
 def test_domain_overflow_and_nonisometry_rejected(self):
  for u,sector,offset in ((np.eye(2),65536,0),(np.eye(2),0,2**32),(np.ones((2,2)),0,0)):
   _,meta=sample(self.lib,u,fock_core(0),.5,2,0x92000114,sector=sector,offset=offset);self.assertEqual(meta['returncode'],2)
