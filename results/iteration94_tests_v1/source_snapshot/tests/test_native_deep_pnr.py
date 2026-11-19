import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_deep_pnr import NativeDeepPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_DEEP_PNR_LIBRARY'),'explicit deep PNR library required')
class DeepPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeDeepPNRSampler(kwargs.pop('v',np.ones((1,1),complex)),kwargs.pop('t',[1/3]),os.environ['GBSLAB_DEEP_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_deep_archived_worker_replay(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');archive=np.load(ROOT/'results/iteration75_deep_pilot_v1/counts_1_7.npz')['counts']
        with self.model(v=d['v'],t=d['t'],retained=7) as model:
            a=model.sample(2048,751408,workers=1);b=model.sample(2048,751408,workers=16)
        np.testing.assert_array_equal(a['counts'],archive[:2048]);np.testing.assert_array_equal(a['counts'],b['counts']);self.assertEqual(a['statistics']['photons'],int(a['counts'].sum()))
    def test_retention_does_not_cap_output_photons(self):
        with self.model(t=[np.tanh(1.1)],retained=7,workers=16) as model:r=model.sample(5000,755001)
        self.assertGreater(int(r['counts'].max()),8);self.assertEqual(r['statistics']['completed_samples'],5000)
    def test_coherent_poisson(self):
        with self.model(t=[0.],mean=[2.,0.],retained=7,workers=16) as model:r=model.sample(50000,755101)
        x=r['counts'][:,0];self.assertLess(abs(x.mean()-2),.04);self.assertLess(abs(x.var()-2),.08)
    def test_empty_vacuum_and_closed(self):
        with self.model(t=[0.],retained=7) as model:
            self.assertFalse(model.sample(100,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,1))
        with self.assertRaises(RuntimeError):model.sample(1,0)
    def test_invalid_depth(self):
        for depth in (True,-1,8,1.5):
            with self.assertRaises(ValueError):self.model(retained=depth)
