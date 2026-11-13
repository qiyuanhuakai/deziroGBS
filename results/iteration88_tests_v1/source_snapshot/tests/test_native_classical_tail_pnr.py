import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_classical_tail_pnr import NativeClassicalTailPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_CLASSICAL_TAIL_PNR_LIBRARY'),'explicit classical completion library required')
class ClassicalTailPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeClassicalTailPNRSampler(np.ones((1,1),complex),kwargs.pop('t',[1/3]),os.environ['GBSLAB_CLASSICAL_TAIL_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_classical_completion_counts_and_worker_replay(self):
        with self.model(retained=0) as model:
            a=model.sample(10000,742101,workers=1);b=model.sample(10000,742101,workers=16)
        np.testing.assert_array_equal(a['counts'],b['counts']);self.assertGreater(a['statistics']['classical_completions'],0);self.assertGreater(a['statistics']['classical_photons'],0);self.assertEqual(a['counts'].sum(),a['statistics']['photons'])
        data=np.load(ROOT/'results/iteration74_classical_pilot_v1/counts_pure_0.npz');np.testing.assert_array_equal(a['counts'],data['counts'][:10000])
    def test_coherent_poisson_target(self):
        with self.model(t=[0.],mean=[2.,0.],retained=0,workers=16) as model:r=model.sample(100000,744201)
        counts=r['counts'][:,0];self.assertLess(abs(counts.mean()-2),.025);self.assertLess(abs(counts.var()-2),.05);self.assertGreater(r['statistics']['classical_completions'],80000)
    def test_empty_and_invalid_displacement(self):
        with self.model() as model:
            self.assertEqual(len(model.sample(0,0)['statistics']),12);model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(1,0)
