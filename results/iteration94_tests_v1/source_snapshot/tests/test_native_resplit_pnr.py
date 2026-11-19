import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_resplit_pnr import NativeResplitPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_RESPLIT_PNR_LIBRARY'),'explicit resplit PNR library required')
class ResplitPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeResplitPNRSampler(kwargs.pop('v',np.ones((1,1),complex)),kwargs.pop('t',[1/3]),os.environ['GBSLAB_RESPLIT_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_resplit_archived_worker_replay(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');archive=np.load(ROOT/'results/iteration75_resplit_pilot_v1/counts_1_7.npz')['counts'];seed=0x78000000000000+7*2**40
        with self.model(v=d['v'],t=d['t'],retained=7) as model:
            a=model.sample(2048,seed,workers=1);b=model.sample(2048,seed,workers=16)
        np.testing.assert_array_equal(a['counts'],archive[:2048]);np.testing.assert_array_equal(a['counts'],b['counts']);self.assertGreater(a['statistics']['resplit_successes'],0);self.assertLessEqual(a['statistics']['resplit_successes'],a['statistics']['resplit_attempts']);self.assertEqual(a['statistics']['photons'],int(a['counts'].sum()))
    def test_coherent_poisson(self):
        with self.model(t=[0.],mean=[2.,0.],retained=0,workers=16) as model:r=model.sample(50000,755201)
        x=r['counts'][:,0];self.assertLess(abs(x.mean()-2),.04);self.assertLess(abs(x.var()-2),.08)
    def test_vacuum_and_empty_statistics(self):
        with self.model(t=[0.],retained=7) as model:
            self.assertFalse(model.sample(100,0)['counts'].any());self.assertEqual(len(model.sample(0,0)['statistics']),12)
    def test_invalid_depth_and_displacement(self):
        with self.assertRaises(ValueError):self.model(retained=8)
        with self.model() as model:
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(1,0)
