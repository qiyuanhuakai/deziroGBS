"""The Windows precision regression behind rejected coherent-core v1."""
import os,unittest,ctypes as C,shutil
from pathlib import Path
import numpy as np
class Precision92Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_COHERENT92_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build coherent92 explicitly')
  cls.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));cls.lib=C.CDLL(path)
 def test_extended_precision_and_restoration(self):
  fn=self.lib.coherent92_precision;fn.argtypes=[C.POINTER(C.c_double)];v=np.zeros(5);fn(v.ctypes.data_as(C.POINTER(C.c_double)));self.assertEqual(v[0],v[4]);self.assertEqual(int(v[2])&0xf00,0x300);self.assertEqual(v[3],2**-60)
