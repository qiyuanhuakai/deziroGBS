"""Independent moment inversion, analytic thermal laws, rejection and replay."""
import ctypes as C, os, shutil, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from iteration88_project_common_v1 import projected, sample, NativeStableClickSampler, DP, ptr

class NativeProjected88Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  value=os.environ.get('GBSLAB_PROJECTED88_LIBRARY')
  if not value or not Path(value).is_file():raise unittest.SkipTest('Build iteration88 projection first')
  cls.dll=Path(value);cls.dirs=[Path(shutil.which('g++')).parent]
 def make(self,v,x):
  model=NativeStableClickSampler(v,x,self.dll,branch_budget=128,runtime_dirs=self.dirs);self.addCleanup(model.close);return model
 def probe(self,model,T,F):
  T=np.ascontiguousarray(T);F=np.ascontiguousarray(F);output=np.zeros_like(T);stats=np.zeros(2);err=C.create_string_buffer(4096);fn=model.lib.tc88_project_probe;fn.argtypes=[C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int];rc=fn(len(T),ptr(T),ptr(F),ptr(output),ptr(stats),err,len(err));return rc,output,stats,err.value
 def test_inverse_preserves_entire_second_moment(self):
  # Deliberately correlated directions outside the retained detector plane.
  import mpmath as mp
  with mp.workdps(70):
   Y=mp.matrix([[.8,.12,.08,.02],[.12,.6,.03,-.05],[.08,.03,.9,.11],[.02,-.05,.11,.7]]);F=mp.matrix([[.7,0],[0,.6],[.2,.1],[-.1,.2]]);P=Y*F;G=mp.eye(2)+F.T*P;q=1/mp.sqrt(mp.det(G));T=Y+q/(1-q)*P*G**-1*P.T
   model=self.make(np.eye(2,dtype=complex),np.eye(4));target=np.array(T.tolist(),float);target=(target+target.T)/2;rc,base,stats,msg=self.probe(model,target,np.array(F.tolist(),float));self.assertEqual(rc,0,msg);self.assertEqual(stats[0],0);np.testing.assert_allclose(base,np.array(Y.tolist(),float),rtol=2e-13,atol=2e-14)
   B=mp.matrix(base.tolist());P=B*F;G=mp.eye(2)+F.T*P;q=1/mp.sqrt(mp.det(G));recovered=B+q/(1-q)*P*G**-1*P.T;self.assertLess(float(max(abs(recovered[i,j]-mp.mpf(target[i,j])) for i in range(4) for j in range(4))),2e-14)
 def test_infeasible_and_invalid_targets_are_explicit(self):
  model=self.make(np.eye(2,dtype=complex),np.eye(4));F=np.eye(4)[:,:2].copy();T=np.diag([-.2,.3,.4,.5]);rc,_,stats,msg=self.probe(model,T,F);self.assertEqual(rc,0,msg);self.assertEqual(stats[0],1)
  T=np.eye(4);T[0,1]=.1;rc,_,_,_=self.probe(model,T,F);self.assertNotEqual(rc,0)
  T=np.eye(4);T[0,0]=np.nan;rc,_,_,_=self.probe(model,T,F);self.assertNotEqual(rc,0)
 def test_independent_thermal_future_and_budget_one(self):
  n=np.linspace(.2,.9,8);model=self.make(np.eye(8,dtype=complex),np.diag(np.r_[n,n]));draws=np.array([[0. if label>>j&1 else np.nextafter(1.,0.) for j in range(8)] for label in range(256)])
  for budget in (1,2,8,64,128):
   raw,meta=projected(model,draws,budget);self.assertEqual(meta['returncode'],0,meta);np.testing.assert_allclose(raw['conditional'],np.tile(n/(1+n),(256,1)),atol=2e-14,rtol=2e-13);s=raw['statistics'];np.testing.assert_array_equal(s[:,12],s[:,13]+s[:,14]);np.testing.assert_array_equal(s[:,14],s[:,16:20].sum(1));self.assertLess(s[:,15].max(),1e-11)
   if budget==1:
    old,om=sample(model,draws,budget);self.assertEqual(om['returncode'],0);np.testing.assert_array_equal(raw['clicks'],old['clicks']);np.testing.assert_array_equal(raw['statistics'][:,:12],old['statistics'])
 def test_threads_chunks_and_numeric_guard(self):
  rng=np.random.default_rng(880104);v=np.linalg.qr(rng.normal(size=(12,3))+1j*rng.normal(size=(12,3)))[0];t=np.array([.3,.5,.7]);model=self.make(v,.8*np.diag(np.r_[-t/(1+t),t/(1-t)]));draws=rng.random((17,12));whole,meta=projected(model,draws,2,threads=4);self.assertEqual(meta['returncode'],0,meta);left,lm=projected(model,draws[:5],2,threads=1);right,rm=projected(model,draws[5:],2,threads=2)
  for m in (lm,rm):self.assertEqual(m['returncode'],0,m)
  for key in whole:np.testing.assert_array_equal(whole[key],np.concatenate([left[key],right[key]]))
  numeric,nm=projected(model,draws,2,guarded=0);self.assertEqual(nm['returncode'],0,nm);np.testing.assert_array_equal(whole['clicks'],numeric['clicks']);np.testing.assert_array_equal(whole['statistics'][:,12:],numeric['statistics'][:,12:])
 def test_invalid_record_is_retained_without_poisoning_neighbor(self):
  model=self.make(np.eye(2,dtype=complex),np.eye(4)*.2);raw,meta=projected(model,np.array([[.5,np.nan],[.2,.3]]),16);self.assertNotEqual(meta['returncode'],0);self.assertEqual(raw['errors'][0,1],1);self.assertNotEqual(raw['errors'][0,0],0);self.assertEqual(raw['errors'][1,0],0);self.assertEqual(raw['statistics'][1,5],2);self.assertEqual(raw['kind'][0,1],255)
