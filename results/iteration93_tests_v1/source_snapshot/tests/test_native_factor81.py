import ctypes as C
import os
import unittest
import numpy as np

DP=C.POINTER(C.c_double)
def ptr(a):return a.ctypes.data_as(DP)

class Factor81Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  name=os.environ.get('GBSLAB_FACTOR81_LIBRARY')
  if not name:raise unittest.SkipTest('Factor81 DLL not configured')
  cls.lib=C.CDLL(name);cls.fn=cls.lib.factor81_ipf
  cls.fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int];cls.fn.restype=C.c_int
 def run_ipf(self,w,p,q,maximum=64,m=None,latent=None):
  w,p,q=[np.ascontiguousarray(a,dtype=float) for a in (w,p,q)];L,M=p.shape;k=np.zeros((M,M));a=np.zeros_like(p);b=a.copy();stats=np.zeros(6);err=C.create_string_buffer(2048)
  rc=self.fn(M if m is None else m,L if latent is None else latent,ptr(w),ptr(p),ptr(q),maximum,1e-10,1,ptr(k),ptr(a),ptr(b),ptr(stats),err,len(err))
  return rc,err.value.decode(),k,a,b,stats
 def test_exact_product_solution(self):
  p=np.array([[.1,.2,.3,.4],[.4,.3,.2,.1]]);w=np.array([.25,.75]);q=np.einsum('l,li,lj->ij',w,p,p)
  rc,msg,k,a,b,s=self.run_ipf(w,p,q,256);self.assertEqual(rc,0,msg);table=k[None]*a[:,:,None]*b[:,None,:];table=(table+table.transpose(0,2,1))/2
  np.testing.assert_allclose(table.sum(2),p,atol=1e-10,rtol=0);np.testing.assert_allclose(np.einsum('l,lij->ij',w,table),q,atol=1e-10,rtol=0)
 def test_incompatible_constraints_fail(self):
  rc,msg,*_=self.run_ipf([1],[[.8,.2]],np.full((2,2),.25));self.assertNotEqual(rc,0);self.assertIn('converge',msg)
 def test_asymmetric_target_rejected(self):
  rc,msg,*_=self.run_ipf([1],[[.5,.5]],[[.1,.2],[.3,.4]]);self.assertNotEqual(rc,0);self.assertIn('probabilities',msg)
 def test_negative_weight_rejected(self):
  rc,msg,*_=self.run_ipf([-1],[[.5,.5]],np.full((2,2),.25));self.assertNotEqual(rc,0);self.assertIn('positive',msg)
 def test_unnormalized_marginal_rejected(self):
  rc,msg,*_=self.run_ipf([1],[[.5,.6]],np.full((2,2),.25));self.assertNotEqual(rc,0);self.assertIn('normalization',msg)
 def test_resource_limit_before_pointer_read(self):
  rc,msg,*_=self.run_ipf([1],[[1]],[[1]],m=1024,latent=131072);self.assertNotEqual(rc,0);self.assertIn('work limit',msg)
 def test_structural_zero_support(self):
  rc,msg,k,a,b,s=self.run_ipf([1],[[1.,0.]],[[1.,0.],[0.,0.]]);self.assertEqual(rc,0,msg);np.testing.assert_array_equal(k*a[0,:,None]*b[0,None,:],[[1,0],[0,0]])

class Histogram81Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  name=os.environ.get('GBSLAB_JZ81_HIST_LIBRARY')
  if not name:raise unittest.SkipTest('Jiuzhang histogram DLL not configured')
  cls.lib=C.CDLL(name);cls.fn=cls.lib.jz81_hist;cls.fn.argtypes=[C.POINTER(C.c_uint8),C.c_uint64,C.c_int,C.POINTER(C.c_int),C.c_uint64,C.c_int,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)];cls.fn.restype=C.c_int
 def invoke(self,data,ports,pilot=1,threads=1):
  data=np.ascontiguousarray(data,dtype=np.uint8);ports=np.ascontiguousarray(ports,dtype=np.int32);out=np.zeros((len(ports),2,256),np.uint64);tot=np.zeros(1153,np.uint64)
  rc=self.fn(data.ctypes.data_as(C.POINTER(C.c_uint8)),len(data),len(ports),ports.ctypes.data_as(C.POINTER(C.c_int)),pilot,threads,out.ctypes.data_as(C.POINTER(C.c_uint64)),tot.ctypes.data_as(C.POINTER(C.c_uint64)));return rc,out,tot
 def test_msb_and_pilot_split(self):
  d=np.zeros((4,144),np.uint8);d[0,0]=128;d[1,0]=1;d[2,0]=255;d[3,-1]=1;p=[list(range(8)),list(range(1144,1152))];rc,h,t=self.invoke(d,p)
  self.assertEqual(rc,0);self.assertEqual(h[0,0,1],1);self.assertEqual(h[0,1,128],1);self.assertEqual(h[0,1,255],1);self.assertEqual(h[1,1,128],1);self.assertEqual(t[1],3);self.assertEqual(t[8],1)
 def test_parallel_histogram_matches_independent_unpack(self):
  d=np.random.default_rng(810081).integers(0,256,(513,144),dtype=np.uint8);p=[list(range(0,1152,144))];rc,h,t=self.invoke(d,p,17,4);self.assertEqual(rc,0);bits=np.unpackbits(d,axis=1);v=(bits[:,p[0]]*(1<<np.arange(8))).sum(1)
  np.testing.assert_array_equal(h[0,0],np.bincount(v[:17],minlength=256));np.testing.assert_array_equal(h[0,1],np.bincount(v[17:],minlength=256));np.testing.assert_array_equal(t,np.bincount(bits.sum(1).astype(int),minlength=1153))
 def test_duplicate_detector_rejected(self):self.assertEqual(self.invoke(np.zeros((2,144)),[[0]*8])[0],3)
 def test_out_of_range_detector_rejected(self):self.assertEqual(self.invoke(np.zeros((2,144)),[[1152]+list(range(7))])[0],2)
 def test_empty_holdout_rejected(self):self.assertNotEqual(self.invoke(np.zeros((2,144)),[list(range(8))],2)[0],0)
