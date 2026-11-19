"""Parity moments, analytic inverse and explicit unsupported boundary states."""
import ctypes as C,os,shutil,unittest
from pathlib import Path
import numpy as np
class Parity90Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_PARITY90_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build parity90 explicitly')
  cls.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));cls.lib=C.CDLL(path);cls.fn=cls.lib.parity90_probe;cls.DP=C.POINTER(C.c_double);cls.fn.argtypes=[C.c_int,C.c_int,C.c_int,cls.DP,cls.DP,cls.DP,C.c_char_p,C.c_int]
 def call(self,x,p,inverse):
  x=np.ascontiguousarray(x);y=np.zeros_like(x);stats=np.zeros(5);err=C.create_string_buffer(4096);rc=self.fn(len(x),p,inverse,x.ctypes.data_as(self.DP),y.ctypes.data_as(self.DP),stats.ctypes.data_as(self.DP),err,len(err));return rc,y,stats,err.value
 def test_thermal_parity_laws_and_fock_moment_inverse(self):
  for n in (.1,1.,3.):
   for p in (1,-1):
    rc,y,_,err=self.call(np.eye(2)*n,p,0);self.assertEqual(rc,0,err);np.testing.assert_allclose(y,np.eye(2)*(2*n*n/(1+2*n)+(p==-1)),atol=2e-13,rtol=0)
  for k in (2,3,4,6,8):
   rc,y,stats,err=self.call(np.eye(2)*k,(-1)**k,1);self.assertEqual(rc,0,err);h=k-k%2;np.testing.assert_allclose(y,np.eye(2)*(h+np.sqrt(h*h+2*h))/2,atol=3e-12,rtol=0);self.assertLess(stats[2],2e-11)
 def test_symplectic_covariance_roundtrip(self):
  S=np.diag([2.,.5,.5,2.]);x=S@np.diag([.8,1.2,.8,1.2])@S.T-np.eye(4)/2
  for p in (1,-1):
   rc,y,_,err=self.call(x,p,0);self.assertEqual(rc,0,err);rc,z,_,err=self.call(y,p,1);self.assertEqual(rc,0,err);np.testing.assert_allclose(z,x,atol=3e-12,rtol=3e-12)
 def test_boundary_and_nonphysical_inputs_are_not_clipped(self):
  for x,p,inverse in ((np.zeros((2,2)),-1,0),(np.eye(2),-1,1),(np.eye(2)*.9,-1,1),(np.diag([-.1,3.,-.1,3.]),1,0)):
   rc,_,_,err=self.call(x,p,inverse);self.assertNotEqual(rc,0,err)
