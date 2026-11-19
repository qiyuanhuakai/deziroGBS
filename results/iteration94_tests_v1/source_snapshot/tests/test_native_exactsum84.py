"""Cancellation, range and rejection tests for the integer dot accumulator."""
import ctypes as C
from decimal import Decimal,localcontext
import os,random,unittest
from pathlib import Path

class ExactSum84Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_EXACTSUM84_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build exactsum84 explicitly')
  cls.runtime=[os.add_dll_directory(p) for p in os.environ.get('GBSLAB_NATIVE_RUNTIME_DIRS','').split(os.pathsep) if p];cls.lib=C.CDLL(path);cls.fn=cls.lib.sum84_probe;cls.fn.argtypes=[C.c_int,C.c_char_p,C.c_char_p,C.c_int,C.c_char_p,C.c_int];cls.fn.restype=C.c_int
 @classmethod
 def tearDownClass(cls):
  for h in cls.runtime:h.close()
 def probe(self,rows):
  data=' '.join(str(v) for row in rows for v in row).encode();out=C.create_string_buffer(2048);err=C.create_string_buffer(4096);rc=self.fn(len(rows),data,out,len(out),err,len(err))
  if rc:raise ValueError(err.value.decode())
  hi,lo,fallback=out.value.decode().split();return Decimal(hi)+Decimal(lo),int(fallback)
 def test_empty_and_exact_zero(self):
  with localcontext() as ctx:
   ctx.prec=200;self.assertEqual(self.probe([]),(0,0));self.assertEqual(self.probe([(2**112-1,1,2**63-1),(-2**112+1,-1,2**63-1)]),(0,0))
 def test_large_cancellation_retains_unit_residual(self):
  with localcontext() as ctx:
   ctx.prec=200;actual,fallback=self.probe([(2**200,1,1),(-2**200,0,1)]);self.assertEqual((actual,fallback),(1,0))
 def test_all_256_products_propagate_carry(self):
  with localcontext() as ctx:
   ctx.prec=200;row=(2**112-1,0,2**64-1);actual,fallback=self.probe([row]*256);self.assertEqual(actual,Decimal(row[0])*row[2]*256);self.assertEqual(fallback,0)
 def test_bounded_exponent_random_dots_and_cancellation(self):
  rng=random.Random(840909)
  with localcontext() as ctx:
   ctx.prec=300
   for case in range(36):
    rows=[]
    for j in range(32):
     hi=Decimal((rng.getrandbits(112)|1)*rng.choice([-1,1]))*Decimal(2)**rng.randrange(-112,-104);lo=Decimal((rng.getrandbits(112)|1)*rng.choice([-1,1]))*Decimal(2)**rng.randrange(-231,-223);x=Decimal((rng.getrandbits(64)|1)*rng.choice([-1,1]))*Decimal(2)**rng.randrange(-70,-62);rows.append((hi,lo,x))
    if case%2==0:rows+= [(-a,-b,c) for a,b,c in rows[:-1]];rng.shuffle(rows)
    expected=sum((a+b)*c for a,b,c in rows);actual,fallback=self.probe(rows);self.assertEqual(fallback,0);self.assertLessEqual(abs(actual-expected),abs(expected)*Decimal(2)**-215)
 def test_wide_exponent_span_explicitly_falls_back(self):
  with localcontext() as ctx:
   ctx.prec=300;actual,fallback=self.probe([(2**500,0,1),(1,0,1)]);self.assertEqual(fallback,1);self.assertLessEqual(abs(actual-(Decimal(2)**500+1)),Decimal(2)**290)
 def test_nonfinite_and_trailing_tokens_are_rejected(self):
  out=C.create_string_buffer(2048);err=C.create_string_buffer(4096)
  for value in (b'nan 0 1',b'1 0 inf',b'1 0 1 extra'):
   self.assertNotEqual(self.fn(1,value,out,len(out),err,len(err)),0)
