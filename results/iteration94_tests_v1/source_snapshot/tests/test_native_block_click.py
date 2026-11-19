import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_block_click import NativeBlockClickSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_BLOCK_CLICK_LIBRARY'),'explicit block library required')
class BlockClickTests(unittest.TestCase):
    def model(self,**kw):
        d=np.load(ROOT/'results/iteration79_pilot_v1/click_input_1.npz')
        return NativeBlockClickSampler(kw.pop('v',d['v']),kw.pop('x',d['x']),os.environ['GBSLAB_BLOCK_CLICK_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kw)
    def test_full_small_law_and_archived_probabilities(self):
        target=np.load(ROOT/'results/iteration79_pilot_v1/click_input_1.npz')['target']
        with self.model(block_size=4,branch_budget=16) as model:law=np.array([model.prefix_probability([(s>>j)&1 for j in range(8)])['probability'] for s in range(256)])
        np.testing.assert_allclose(law,target,atol=1e-12,rtol=0)
    def test_partial_prefix_consistency(self):
        with self.model(block_size=4,branch_budget=1) as model:
            for bits in ([],[1],[1,0],[1,0,1],[1,0,1,0],[1,0,1,0,1]):
                p=model.prefix_probability(bits)['probability'];q=sum(model.prefix_probability(bits+[b])['probability'] for b in (0,1));self.assertAlmostEqual(p,q,places=12)
    def test_workers_replay_and_zero_count(self):
        with self.model(block_size=4,branch_budget=4) as model:a=model.sample(257,791101,workers=1);b=model.sample(257,791101,workers=16);empty=model.sample(0,0)
        np.testing.assert_array_equal(a['clicks'],b['clicks']);self.assertEqual(empty['clicks'].shape,(0,8))
    def test_noise_requires_realized_mean(self):
        with self.model(noise_variance=np.full(8,.1)) as model:
            with self.assertRaises(ValueError):model.prefix_probability([0])
            self.assertGreater(model.prefix_probability([0],mean=np.zeros(8))['probability'],0)
    def test_invalid_width(self):
        for width in (0,5,True,2.0):
            with self.assertRaises(ValueError):self.model(block_size=width)
