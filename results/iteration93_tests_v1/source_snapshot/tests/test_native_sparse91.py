"""Sparse zero-pivot suffix regression: preserve trailing optical components."""
import ctypes as C,os,shutil,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration91_common_v1 import optical_cache,NativeStableClickSampler,ptr,DP
class Sparse91Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_OPTICAL91_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build corrected compact91')
  cls.dll=Path(path);cls.dirs=[Path(shutil.which('g++')).parent]
 def check_map(self,v):
  with NativeStableClickSampler(v,np.eye(2*v.shape[1])*.2,self.dll,branch_budget=16,runtime_dirs=self.dirs) as model:
   h,d,meta=optical_cache(model)
   try:
    self.assertLess(meta['gram_residual'],1e-15);fn=model.lib.tc91_optical_export;fn.argtypes=[C.c_void_p,C.c_int,DP,C.c_char_p,C.c_int]
    for j in range(len(v)):
     k=min(v.shape[1],len(v)-j);raw=np.zeros((k,v.shape[1],2));msg=C.create_string_buffer(4096);rc=fn(h,j,ptr(raw),msg,len(msg));self.assertEqual(rc,0,msg.value);a=raw[:,:,0]+1j*raw[:,:,1];np.testing.assert_allclose(a.conj().T@a,v[j:].conj().T@v[j:],atol=2e-15,rtol=2e-15)
   finally:d(h)
 def test_identity_short_suffix_has_leading_zeros(self):self.check_map(np.eye(6,dtype=complex))
 def test_permuted_complex_axes(self):
  v=np.eye(6,dtype=complex)[[5,1,4,2,0,3]];v*=np.exp(1j*np.arange(6))[None,:];self.check_map(v)
 def test_interleaved_zero_rows(self):
  v=np.zeros((12,5),complex);v[[0,2,5,8,11]]=np.eye(5);self.check_map(v)
 def test_block_unitary_with_zero_pivots(self):
  u=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2);v=np.zeros((6,6),complex)
  for j in range(3):v[2*j:2*j+2,2*j:2*j+2]=u
  self.check_map(v)
