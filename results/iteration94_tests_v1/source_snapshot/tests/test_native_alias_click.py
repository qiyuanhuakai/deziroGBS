import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_alias_click import NativeAliasClickSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_ALIAS_CLICK_LIBRARY'),'explicit alias library required')
class AliasClickTests(unittest.TestCase):
    def model(self,cov=None,**kwargs):return NativeAliasClickSampler(np.eye(8) if cov is None else cov,os.environ['GBSLAB_ALIAS_CLICK_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_independent_complete_laws(self):
        for case in (2,3):
            d=np.load(ROOT/f'results/iteration78_pilot_v2/table_{case}.npz')
            with self.model(d['cov']) as model:law=model.law()
            self.assertLess(abs(law['probabilities']-d['target']).sum()/2,1e-12);self.assertAlmostEqual(law['raw_total'],1,places=13)
    def test_worker_and_chunk_replay(self):
        d=np.load(ROOT/'results/iteration78_pilot_v2/table_2.npz')
        with self.model(d['cov']) as model:
            a=model.sample(4096,781101,workers=1)['clicks'];b=model.sample(4096,781101,workers=16)['clicks'];c=np.concatenate([model.sample(1024,781101,row_offset=i*1024)['clicks'] for i in range(4)])
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
    def test_thermal_product_law_at_resource_ceiling(self):
        m=16;p=np.linspace(.12,.4,m);cov=np.diag(np.r_[(1+p)/(1-p),(1+p)/(1-p)]);states=np.arange(1<<m,dtype=np.uint32);target=np.prod(np.array([np.where(states>>j&1,p[j],1-p[j]) for j in range(m)]),axis=0)
        with self.model(cov,workers=1) as model:law=model.law()['probabilities'];r=model.sample(8192,781102)
        self.assertLess(abs(law-target).sum()/2,1e-10);self.assertLess(abs(r['clicks'].mean(axis=0)-p).max(),.025)
    def test_vacuum_empty_and_closed(self):
        with self.model() as model:self.assertFalse(model.sample(100,0)['clicks'].any());self.assertEqual(model.sample(0,0)['clicks'].shape,(0,4));self.assertEqual(model.law()['probabilities'][0],1.)
        with self.assertRaises(RuntimeError):model.sample(1,0)
    def test_invalid_inputs_and_negative_mass_not_clipped(self):
        for cov in (np.eye(34),np.eye(3),np.full((2,2),np.nan)):
            with self.assertRaises(ValueError):self.model(cov)
        with self.assertRaisesRegex(ArithmeticError,'Negative Mobius'):self.model(np.eye(2)*.5)
        with self.model() as model:
            with self.assertRaises(ValueError):model.sample(2,0,row_offset=2**64-1)
            with self.assertRaises(ValueError):model.sample(2,0,workers=True)
    def test_asymmetric_covariance_rejected(self):
        cov=np.eye(4);cov[0,1]=.1
        with self.assertRaisesRegex(ArithmeticError,'Asymmetric'):self.model(cov)
