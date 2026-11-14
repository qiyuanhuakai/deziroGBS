"""Adaptive Schmidt rank must charge small discarded amplitudes explicitly."""
import os,sys,unittest,ctypes as C
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration93_common_v5 import load93,ptr,DP

class AdaptiveMps93Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_MPS93_PROBE_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build MPS93 adaptive probe explicitly')
  cls.production,cls.runtime=load93();cls.lib=C.CDLL(path);setter=cls.lib.mps93_set_lapack;setter.argtypes=[C.c_void_p]*3;assert setter(*[C.cast(getattr(cls.production._blas,name),C.c_void_p) for name in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
  cls.fn=cls.lib.mps93_probe;cls.fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_double,DP,C.c_int,C.c_double,DP,DP,DP,DP,C.c_char_p,C.c_int]
 def probe(self,epsilon):
  n,d=3,4;core=np.zeros((d,)*n,complex);core[2,0,0]=np.sqrt(1-epsilon**2);core[1,2,2]=epsilon;t=np.zeros(n);alpha=np.zeros(n,complex);w=np.array([1,0,0],complex);flow=np.zeros_like(core);jump=np.zeros_like(core);gram=np.zeros((n,n),complex);stats=np.zeros(7);error=C.create_string_buffer(4096);rc=self.fn(n,d,ptr(core.view(float)),ptr(t),ptr(alpha.view(float)),.7,ptr(w.view(float)),16,1.,ptr(flow.view(float)),ptr(jump.view(float)),ptr(gram.view(float)),ptr(stats),error,len(error));self.assertEqual(rc,0,error.value);reference=np.zeros_like(core);reference[1,0,0]=np.sqrt(2)*core[2,0,0];reference[0,2,2]=epsilon;reference/=np.linalg.norm(reference);overlap=np.vdot(jump,reference);orthogonal=float(np.linalg.norm(reference-overlap*jump));return jump,stats,orthogonal
 def test_discarded_small_direction_is_charged(self):
  epsilon=3e-14;jump,stats,actual=self.probe(epsilon);self.assertEqual(stats[5],1);self.assertGreater(stats[3],0);np.testing.assert_allclose(stats[3],epsilon/np.sqrt(2),rtol=.02,atol=3e-16);np.testing.assert_allclose(actual,stats[3],rtol=.02,atol=3e-16)
 def test_resolved_direction_is_retained(self):
  jump,stats,actual=self.probe(1e-9);self.assertEqual(stats[5],2);self.assertLess(stats[3],1e-14);self.assertLess(actual,1e-14);self.assertGreater(abs(jump[0,2,2]),5e-10)
