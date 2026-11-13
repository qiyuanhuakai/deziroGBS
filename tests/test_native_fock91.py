"""Conditional finite-occupation core interference and inclusive tails."""
import ctypes as C,os,shutil,sys,unittest,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration91_fock_common_v1 import core_law,core_draw
class Fock91Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_FOCK91_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build fock91 explicitly')
  cls.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));cls.lib=C.CDLL(path)
 def test_identity_negative_binomial_and_charge(self):
  keys,p,meta=core_law(self.lib,np.eye(2,dtype=complex),4,.6,24);self.assertEqual(meta['returncode'],0,meta)
  for key,prob in zip(keys,p):
   n=int(key)&255;m=(int(key)>>8)&255;expected=math.comb(m+4,4)*.64**5*.36**m if n-m==4 else 0;self.assertAlmostEqual(prob,expected,places=13)
  self.assertAlmostEqual(p.sum()+meta['stats'][1],1,places=12)
 def test_hong_ou_mandel_interference(self):
  u=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);keys,p,meta=core_law(self.lib,u,0,.5,2);self.assertEqual(meta['returncode'],0);law=dict(zip(map(int,keys),p));self.assertAlmostEqual(law[257],0,places=14);self.assertAlmostEqual(law[2],.09375,places=13);self.assertAlmostEqual(law[512],.09375,places=13)
 def test_one_photon_boundary_is_represented(self):
  u=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2);keys,p,meta=core_law(self.lib,u,1,0,1);self.assertEqual(meta['returncode'],0);np.testing.assert_allclose(sorted(p),[.5,.5],atol=2e-15);self.assertEqual(meta['stats'][1],0)
 def test_tail_is_failure_without_refill(self):
  keys,p,meta=core_law(self.lib,np.eye(2,dtype=complex),0,.5,0);values,fails,result=core_draw(self.lib,keys,p,np.array([0.,.749,.75,.999]));self.assertEqual(result['returncode'],0);np.testing.assert_array_equal(fails,[0,0,1,1]);self.assertAlmostEqual(meta['stats'][1],.25)
 def test_invalid_optics_and_draw_fail(self):
  _,_,meta=core_law(self.lib,np.ones((2,2),complex),1,.5,4);self.assertNotEqual(meta['returncode'],0)
  _,_,meta=core_draw(self.lib,np.array([0],np.uint64),np.array([.5]),np.array([np.nan]));self.assertNotEqual(meta['returncode'],0)
