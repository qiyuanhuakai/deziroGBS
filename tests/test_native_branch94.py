"""Direct Gaussian threshold state transitions, failure outcomes and stream replay."""
import os,sys,unittest,ctypes as C,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration94_branch_common_v1 import gaussian94,branch_sample94,branch_enumerate94

class GaussianBranch94Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('GBSLAB_BRANCH94_LIBRARY')
  if not path or not Path(path).is_file():raise unittest.SkipTest('Build direct Gaussian-branch94 explicitly')
  cls.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));cls.lib=C.CDLL(path)
 def test_vacuum_consumes_one_click_word_per_port(self):
  raw,meta=branch_sample94(self.lib,np.eye(24),np.zeros(24),32,0x94008000,limit=1,budget=0)
  self.assertEqual(meta['returncode'],0,meta);self.assertFalse(raw['clicks'].any());np.testing.assert_array_equal(raw['statistics'][:,9],12);np.testing.assert_array_equal(raw['statistics'][:,1],1)
 def test_independent_thermal_complete_law(self):
  photons=np.array([.1,.4,.8,1.2]);q=np.diag(np.r_[1+photons,1+photons]);p0=1/(1+photons);target=np.array([np.prod([p0[j] if not (key>>j)&1 else 1-p0[j] for j in range(4)]) for key in range(16)])
  law,stats,_=branch_enumerate94(self.lib,q,np.zeros(8),8,.01);np.testing.assert_allclose(law,np.r_[target,0],atol=2e-14,rtol=0);self.assertEqual(stats[1],0)
 def test_displaced_and_lossy_complete_laws(self):
  for case in (1,3):
   q,mean=gaussian94(np.load(ROOT/f'results/iteration93_reference_v2/input_{case}.npz'));target=np.load(ROOT/f'results/iteration93_reference_v2/law_{case}.npz')['click'];law,stats,_=branch_enumerate94(self.lib,q,mean,8,.01);np.testing.assert_allclose(law,np.r_[target,0],atol=2e-14,rtol=0);self.assertEqual(stats[1],0)
 def test_budget_failures_are_not_replaced(self):
  q,mean=gaussian94(np.load(ROOT/'results/iteration93_reference_v2/input_0.npz'));raw,meta=branch_sample94(self.lib,q,mean,128,0x94008001,limit=1,budget=0)
  self.assertEqual(meta['returncode'],1);self.assertEqual(len(raw['clicks']),128);self.assertGreater(meta['failures'],0);self.assertLess(meta['failures'],128);self.assertTrue(np.isin(raw['errors'],[0,2]).all());np.testing.assert_array_equal(raw['statistics'][:,1],raw['errors']==0);self.assertTrue((raw['statistics'][raw['errors']!=0,0]<4).all());self.assertFalse(raw['statistics'][:,4].any())
 def test_workers_and_offsets_replay_displaced_state(self):
  q,mean=gaussian94(np.load(ROOT/'results/iteration93_reference_v2/input_1.npz'));kwargs=dict(limit=4,budget=1,sector=7);raw,_=branch_sample94(self.lib,q,mean,128,0x94008002,offset=17,**kwargs)
  parts=[branch_sample94(self.lib,q,mean,count,0x94008002,offset=offset,threads=4,**kwargs)[0] for offset,count in ((17,43),(60,85))]
  for key in raw:np.testing.assert_array_equal(raw[key],np.concatenate([part[key] for part in parts]))
 def test_address_and_parameter_bounds_rejected(self):
  for kwargs in (dict(offset=2**32),dict(sector=65536),dict(limit=65),dict(budget=-.1)):
   _,meta=branch_sample94(self.lib,np.eye(4),np.zeros(4),1,0x94008003,**kwargs);self.assertEqual(meta['returncode'],2,meta)
