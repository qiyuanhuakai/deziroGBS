"""Stream-domain regression gates. Imports never compile native libraries."""
import ctypes as C
import os
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
class DomainRng84Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_RNG84_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build and configure domain RNG84 explicitly')
  cls.runtime=[os.add_dll_directory(p) for p in os.environ.get('GBSLAB_NATIVE_RUNTIME_DIRS','').split(os.pathsep) if p];cls.lib=C.CDLL(path);cls.U32=C.POINTER(C.c_uint32);cls.U64=C.POINTER(C.c_uint64);cls.IP=C.POINTER(C.c_int)
  cls.block=cls.lib.rng84_block;cls.block.argtypes=[cls.U32,cls.U32,cls.U32];cls.block.restype=C.c_int
  cls.stream=cls.lib.rng84_stream;cls.stream.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,cls.U64,C.c_char_p,C.c_int];cls.stream.restype=C.c_int
  cls.address=cls.lib.rng84_address;cls.address.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,cls.U32,cls.IP,C.c_char_p,C.c_int];cls.address.restype=C.c_int
 @classmethod
 def tearDownClass(cls):
  for h in cls.runtime:h.close()
 def words(self,experiment=1,sector=0,purpose=1,row=0,start=0,count=16):
  out=np.zeros(max(1,count),np.uint64);err=C.create_string_buffer(4096);rc=self.stream(experiment,sector,purpose,row,start,count,out.ctypes.data_as(self.U64),err,len(err))
  if rc:raise ValueError(err.value.decode())
  return out[:count]
 def addr(self,experiment,sector,purpose,row,draw):
  out=np.zeros(6,np.uint32);lane=np.zeros(1,np.int32);err=C.create_string_buffer(4096);rc=self.address(experiment,sector,purpose,row,draw,out.ctypes.data_as(self.U32),lane.ctypes.data_as(self.IP),err,len(err))
  if rc:raise ValueError(err.value.decode())
  return (*map(int,out),int(lane[0]))
 def test_all_pinned_upstream_known_answers(self):
  checked=0
  for name in ('kat_vectors','old_kat_vectors'):
   for line in (ROOT/'results/iteration84_rng_reference_v1'/name).read_text().splitlines():
    words=line.split()
    if words[:2]!=['philox4x32','10']:continue
    a=[int(w,16) for w in words[2:]];ctr=np.array(a[:4],np.uint32);key=np.array(a[4:6],np.uint32);out=np.zeros(4,np.uint32);self.assertEqual(self.block(ctr.ctypes.data_as(self.U32),key.ctypes.data_as(self.U32),out.ctypes.data_as(self.U32)),0);self.assertEqual(out.tolist(),a[6:]);checked+=1
  self.assertEqual(checked,55)
 def test_experiment_sector_record_purpose_and_lane_are_injective(self):
  seen=set()
  for experiment in (0,1,1<<32,2**64-1):
   for sector in (0,1,65535):
    for purpose in (1,2,3,4,5,65535):
     for row in (0,1,1023,2**32-1):
      for draw in (0,1,2,3,2**33,2**64-2):
       a=self.addr(experiment,sector,purpose,row,draw);self.assertNotIn(a,seen);seen.add(a);self.assertEqual(a,(experiment&0xffffffff,experiment>>32,(draw//2)&0xffffffff,(draw//2)>>32,row,(sector<<16)|purpose,draw%2))
 def test_reject_field_overflow_instead_of_truncation(self):
  for sector,purpose,row in ((65536,1,0),(0,65536,0),(0,1,2**32)):
   with self.assertRaisesRegex(ValueError,'overflow'):self.words(sector=sector,purpose=purpose,row=row)
 def test_counter_exhaustion_cannot_wrap(self):
  self.assertEqual(len(self.words(start=2**64-2,count=1)),1)
  with self.assertRaisesRegex(ValueError,'exhausted'):self.words(start=2**64-1,count=1)
  with self.assertRaisesRegex(ValueError,'exhausted'):self.words(start=2**64-2,count=2)
 def test_odd_and_even_continuations_preserve_words(self):
  whole=self.words(count=32)
  for start in (0,1,7,16,23):np.testing.assert_array_equal(self.words(start=start,count=8),whole[start:start+8])
 def test_stream_words_match_independently_addressed_blocks(self):
  words=self.words(experiment=2**63+7,sector=17,purpose=4,row=92,count=12)
  for draw in range(12):
   a=self.addr(2**63+7,17,4,92,draw);ctr=np.array(a[2:6],np.uint32);key=np.array(a[:2],np.uint32);out=np.zeros(4,np.uint32);self.assertEqual(self.block(ctr.ctypes.data_as(self.U32),key.ctypes.data_as(self.U32),out.ctypes.data_as(self.U32)),0);lane=a[6]*2;self.assertEqual(int(words[draw]),int(out[lane])|(int(out[lane+1])<<32))
 def test_zero_batch_consumes_no_words(self):
  self.assertEqual(self.words(count=0).shape,(0,));np.testing.assert_array_equal(self.words(count=1),self.words(count=1))
