from pathlib import Path
import os,unittest,shutil,ctypes as C
import numpy as np
from gbslab.native_factor_pnr import NativeFactorPNRSampler
from gbslab.native_stable_click import NativeStableClickSampler
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_direct_click import DP
ROOT=Path(__file__).resolve().parents[1]

class FactorSampling82Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.dll=os.environ.get('GBSLAB_FACTOR82_LIBRARY')
  if not cls.dll:raise unittest.SkipTest('Factor82 DLL not configured')
  cls.runtime=[Path(shutil.which('g++')).parent];d=np.load(ROOT/'results/iteration81_factor_pilot_v2/input_strong4.npz');f=np.load(ROOT/'results/iteration81_factor_pilot_v2/factors_strong4_0.npz');cls.args=dict(v=d['v'],squeezing=np.tanh(1.1),library=cls.dll,kernel=d['K'],left=f['left'],right=f['right'],latent_weights=d['weights'],runtime_dirs=cls.runtime)
 def make(self,**kwargs):return NativeFactorPNRSampler(**{**self.args,**kwargs})
 def test_exact_low_branch_historical_replay(self):
  with self.make(exact_total_budget=64) as m:r=m.sample(128,0x88000000000000+2**40)
  old=np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts'][:128];np.testing.assert_array_equal(r['counts'],old)
 def test_parallel_replay_and_fixed_counts(self):
  with self.make() as m:a=m.sample(257,820101,workers=1,fixed_total=34);b=m.sample(257,820101,workers=16,fixed_total=34)
  np.testing.assert_array_equal(a['counts'],b['counts']);np.testing.assert_array_equal(a['counts'].sum(1),np.full(257,34));self.assertGreater(a['statistics']['pair_rejections'],0)
 def test_actual_law_matches_calibrated_factors(self):
  with self.make() as m:law=m.law()
  p=self.args['kernel'][None]*self.args['left'][:,:,None]*self.args['right'][:,None,:];p/=p.sum((1,2))[:,None,None];self.assertLess(np.max(abs(p-law['oriented_pairs']).sum((1,2))/2),1e-12);np.testing.assert_allclose(law['oriented_pairs'].sum((1,2)),1,atol=1e-14)
 def test_vacuum_and_impossible_fixed_count(self):
  with self.make(squeezing=0.) as m:
   np.testing.assert_array_equal(m.sample(33,10)['counts'],np.zeros((33,4)));self.assertRaises(ArithmeticError,m.sample,1,10,fixed_total=2)
 def test_zero_batch_and_closed_model(self):
  m=self.make();self.assertEqual(m.sample(0,1)['counts'].shape,(0,4));m.close();self.assertRaises(RuntimeError,m.sample,1,1)
 def test_reject_invalid_physical_factors(self):
  with self.assertRaises(ArithmeticError):self.make(kernel=self.args['kernel']*1.01)
 def test_reject_negative_and_wrong_shapes(self):
  with self.assertRaises(ValueError):self.make(left=-self.args['left'])
  with self.assertRaises(ValueError):self.make(left=self.args['left'][:1])
 def test_mutated_noise_cannot_be_ignored(self):
  with self.make() as m:m.noise_variance[0]=.1;self.assertRaises(ValueError,m.sample,1,1)
 def test_reject_invalid_total_and_budget(self):
  with self.assertRaises(ValueError):self.make(exact_total_budget=3)
  with self.make() as m:
   for total in (True,3,-2):self.assertRaises(ValueError,m.sample,1,1,fixed_total=total)
 def test_integer_exports_have_complete_tables(self):
  with self.make() as m:d=m.integer_tables(0)
  self.assertEqual(int(d['sizes'].sum()),len(d['thresholds']));self.assertTrue(np.all(d['accept']<=2**53));self.assertGreater(int(d['accept'].max()),0)

class StableSampling82Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.dll=os.environ.get('GBSLAB_STABLE82_LIBRARY')
  if not cls.dll:raise unittest.SkipTest('Stable82 DLL not configured')
  cls.runtime=[Path(shutil.which('g++')).parent];cls.covs=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/input_0.npz')['covariances']
 def make(self,sector=1,budget=8,**kw):return NativeStableClickSampler(np.eye(8),(self.covs[sector]-np.eye(16))/2,self.dll,branch_budget=budget,runtime_dirs=self.runtime,**kw)
 def test_previously_failing_prefix_is_supported(self):
  old=ROOT/'results/iteration71_branch_pilot_v1/branches.dll'
  with NativeBranchClickSampler(np.eye(8),(self.covs[1]-np.eye(16))/2,old,branch_budget=8,runtime_dirs=self.runtime) as m:self.assertRaises(ArithmeticError,m.prefix_probability,[1,1,1,1,0])
  with self.make() as m:self.assertGreater(m.prefix_probability([1,1,1,1,0])['probability'],0)
 def test_complete_weak_law(self):
  with self.make(budget=32) as m:p=np.array([m.prefix_probability([(i>>j)&1 for j in range(8)])['probability'] for i in range(256)])
  ref=np.load(ROOT/'results/iteration82_stable_pilot_v1/references_0.npz')['sector_laws'][1];self.assertLess(abs(p-ref).sum()/2,1e-14);self.assertAlmostEqual(p.sum(),1,13)
 def test_reproducible_workers(self):
  with self.make(sector=0,budget=16) as m:a=m.sample(129,821001,workers=1);b=m.sample(129,821001,workers=16)
  np.testing.assert_array_equal(a['clicks'],b['clicks']);self.assertEqual(a['statistics']['completed_samples'],129)
 def test_rare_b128_failure_remains_explicit(self):
  with self.make(budget=128) as m:
   with self.assertRaisesRegex(ArithmeticError,'normalization lost'):m.prefix_probability([1,1,1,1,1,1,1,0])
 def test_displacement_and_noise_rejected(self):
  with self.assertRaises(ValueError):self.make(mean=np.ones(16))
  with self.assertRaises(ValueError):self.make(noise_variance=np.ones(16))
  with self.make() as m:m.mean[0]=.1;self.assertRaises(ValueError,m.sample,1,1)
 def test_invalid_resource_controls(self):
  with self.assertRaises(ValueError):self.make(budget=129)
  with self.make() as m:
   self.assertRaises(ValueError,m.sample,1,1,workers=65);self.assertRaises(ValueError,m.prefix_probability,[2]);self.assertRaises(ValueError,m.sample,True,1)
