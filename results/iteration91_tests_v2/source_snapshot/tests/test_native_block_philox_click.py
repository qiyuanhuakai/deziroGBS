import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_block_philox_click import NativeBlockPhiloxClickSampler
from gbslab.native_block_click import NativeBlockClickSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_BLOCK_PHILOX_LIBRARY'),'explicit block GPU library required')
class BlockPhiloxTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(791200);v=np.linalg.qr(rng.normal(size=(5,2))+1j*rng.normal(size=(5,2)))[0];t=.4;x=np.diag([-t/(1+t)]*2+[t/(1-t)]*2);return v,x
    def model(self,**kw):
        v,x=self.data();return NativeBlockPhiloxClickSampler(v,x,ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=os.environ['GBSLAB_BLOCK_PHILOX_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent],**kw)
    def test_partial_final_block_full_law(self):
        v,x=self.data();patterns=np.array([[(s>>j)&1 for j in range(5)] for s in range(32)],np.uint8);draws=np.where(patterns,0.,np.nextafter(1.,0.))
        for width,budget in ((2,1),(4,1),(4,3)):
            with self.model(block_size=width,branch_budget=budget) as model:r=model.replay(np.zeros((32,4)),draws)
            with NativeBlockClickSampler(v,x,os.environ['GBSLAB_BLOCK_CLICK_LIBRARY'],block_size=width,branch_budget=budget,runtime_dirs=[Path(shutil.which('g++')).parent]) as model:oracle=[model.prefix_probability(bits)['probability'] for bits in patterns]
            np.testing.assert_array_equal(r['clicks'],patterns);law=np.prod(np.where(patterns,r['conditional_click_probabilities'],1-r['conditional_click_probabilities']),axis=1);np.testing.assert_allclose(law,oracle,atol=1e-11,rtol=0);self.assertAlmostEqual(law.sum(),1.,places=11)
    def test_row_continuation_and_random_replay(self):
        with self.model(block_size=4,branch_budget=3,noise_variance=np.full(4,.1)) as model:
            a=model.sample(257,791201,row_offset=11,export_random=True);b=model.sample(101,791201,row_offset=11);c=model.sample(156,791201,row_offset=112);r=model.replay(a['realized_means'],a['uniform_draws'])
        np.testing.assert_array_equal(a['clicks'],np.concatenate([b['clicks'],c['clicks']]));np.testing.assert_array_equal(a['conditional_click_probabilities'],r['conditional_click_probabilities'])
    def test_workspace_growth(self):
        with self.model(block_size=1,branch_budget=4) as model:a=model.sample(17,791202)
        with self.model(block_size=4,branch_budget=4) as model:b=model.sample(17,791202)
        self.assertEqual(b['cuda_timings']['workspace_bytes']-a['cuda_timings']['workspace_bytes'],17*(32-4)*(4*4+4*4)*8)
    def test_resource_and_offset_rejection(self):
        with self.assertRaises(ValueError):self.model(block_size=4,branch_budget=17)
        with self.model(block_size=4,branch_budget=4) as model:
            with self.assertRaises(ValueError):model.sample(1,0,branch_budget=17)
            with self.assertRaises(ValueError):model.sample(2,0,row_offset=2**64-1)
            self.assertEqual(model.sample(0,0)['clicks'].shape,(0,5))
