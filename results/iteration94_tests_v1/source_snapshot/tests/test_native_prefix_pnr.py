import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_prefix_pnr import NativePrefixPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_PREFIX_PNR_LIBRARY'),'explicit prefix PNR library required')
class PrefixPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativePrefixPNRSampler(kwargs.pop('v',np.ones((1,1),complex)),kwargs.pop('t',[1/3]),os.environ['GBSLAB_PREFIX_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_archived_worker_replay(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');archive=np.load(ROOT/'results/iteration75_prefix_pilot_v1/counts_1_31.npz')['counts'];seed=0x7b000000000000+7*2**40
        with self.model(v=d['v'],t=d['t'],retained=31) as model:
            a=model.sample(2048,seed,workers=1);b=model.sample(2048,seed,workers=16)
        np.testing.assert_array_equal(a['counts'],archive[:2048]);np.testing.assert_array_equal(a['counts'],b['counts']);self.assertEqual(a['statistics']['photons'],int(a['counts'].sum()))
    def test_same_random_prefix_coupling(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');values=[]
        for k in (7,15,31):
            with self.model(v=d['v'],t=d['t'],retained=k,workers=16) as model:values.append(model.sample(5000,757001)['counts'])
        reference=values[-1]
        for k,counts in zip((7,15),values[:-1]):
            exact_prefix=reference.sum(axis=1)<=k
            np.testing.assert_array_equal(counts[exact_prefix],reference[exact_prefix])
            self.assertTrue(np.all(counts[np.any(counts!=reference,axis=1)].sum(axis=1)>=k+1))
    def test_no_output_cutoff(self):
        d=np.load(ROOT/'results/iteration75_prefix_pilot_v1/counts_1_31.npz')['counts'];self.assertGreater(int(d.sum(axis=1).max()),32)
    def test_vacuum_empty_and_coherent(self):
        with self.model(t=[0.],retained=31) as model:
            self.assertFalse(model.sample(100,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,1))
        with self.model(t=[0.],retained=31,mean=[2.,0.],workers=16) as model:r=model.sample(50000,757101)
        x=r['counts'][:,0];self.assertLess(abs(x.mean()-2),.04);self.assertLess(abs(x.var()-2),.08)
    def test_explicit_depth_and_resource_limits(self):
        for value in (True,-1,32,1.5):
            with self.assertRaises(ValueError):self.model(retained=value)
        with self.model(v=np.eye(8,dtype=complex),t=np.full(8,.2),retained=31) as model:
            with self.assertRaisesRegex(ArithmeticError,'resource estimate'):model.sample(1,0)
