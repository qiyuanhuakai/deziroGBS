import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_philox_click import NativePhiloxClickSampler

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('GBSLAB_PHILOX_CLICK_LIBRARY') and os.environ.get('GBSLAB_CLICK_REPLAY_LIBRARY'),'explicit Philox and CPU replay libraries required')
class PhiloxClickTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativePhiloxClickSampler(kwargs.pop('v',np.eye(2,dtype=complex)),kwargs.pop('x',np.zeros((4,4))),os.environ['GBSLAB_CLICK_REPLAY_LIBRARY'],gpu_library=os.environ['GBSLAB_PHILOX_CLICK_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent],**kwargs)

    def test_vacuum_empty_and_closed(self):
        with self.model() as model:
            r=model.sample(128,736201);self.assertFalse(r['clicks'].any());self.assertEqual(r['statistics']['completed_samples'],128)
            self.assertEqual(model.sample(0,0)['clicks'].shape,(0,2))
            self.assertEqual(model.sample(1,0,row_offset=2**64-1)['clicks'].shape,(1,2))
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_thermal_law_and_full_precision_uniforms(self):
        with self.model(noise_variance=np.full(4,.4)) as model:r=model.sample(30000,736202,export_random=True)
        codes=r['clicks']@np.array([1,2]);hist=np.bincount(codes,minlength=4)/len(codes);p=.4/1.4
        self.assertLess(abs(hist-np.array([(1-p)**2,p*(1-p),p*(1-p),p*p])).sum()/2,.012)
        draws=r['uniform_draws'];self.assertTrue(np.all((draws>=0)&(draws<1)))
        self.assertGreater(np.count_nonzero(np.modf(draws*2**32)[0]),50000)

    def test_batch_boundaries_and_stream_continuation(self):
        with self.model(noise_variance=np.full(4,.4)) as model:
            whole=model.sample(1024,736203,row_offset=123,export_random=True)
            a=model.sample(311,736203,row_offset=123,export_random=True)
            b=model.sample(713,736203,row_offset=434,export_random=True)
        for key in ('clicks','realized_means','uniform_draws','conditional_click_probabilities'):
            np.testing.assert_array_equal(whole[key],np.concatenate([a[key],b[key]]))

    def test_archived_replay_and_exported_inputs(self):
        data=np.load(ROOT/'results/iteration71_branch_pilot_v1/input_4.npz');expected=np.load(ROOT/'results/iteration73_philox_pilot_v1/replay_4_8.npz')
        with self.model(v=data['v'],x=data['x'],mean=data['mean'],branch_budget=8) as model:
            r=model.sample(1024,733004,export_random=True);again=model.replay(r['realized_means'],r['uniform_draws'])
        np.testing.assert_array_equal(r['clicks'],expected['gpu'][:1024]);np.testing.assert_array_equal(r['clicks'],again['clicks'])

    def test_validation_and_failure_recovery(self):
        with self.model() as model:
            for kw in ({'row_offset':-1},{'row_offset':2**64},{'row_offset':2**64-1},{'export_random':1},{'branch_budget':129}):
                with self.assertRaises(ValueError):model.sample(2,0,**kw)
            for count,seed in ((True,0),(-1,0),(1048577,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(1,0)
            model.mean[0]=0;self.assertFalse(model.sample(10,0)['clicks'].any())

if __name__=='__main__':unittest.main()
