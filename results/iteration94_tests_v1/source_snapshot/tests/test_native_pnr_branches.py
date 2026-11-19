import ctypes as C
import os
from pathlib import Path
import shutil
import unittest
import math
import numpy as np
from gbslab.native_pnr_branches import NativeBranchPNRSampler

@unittest.skipUnless(os.environ.get('GBSLAB_BRANCH_PNR_LIBRARY'),'explicit finite-tap PNR library required')
class NativeBranchPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeBranchPNRSampler(kwargs.pop('v',np.eye(2,dtype=complex)),
            kwargs.pop('x',np.zeros((4,4))),os.environ['GBSLAB_BRANCH_PNR_LIBRARY'],
            runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)

    def test_vacuum_and_empty(self):
        with self.model(workers=8) as model:
            result=model.sample(128,716001)
            self.assertFalse(result['counts'].any());self.assertEqual(result['counts'].dtype,np.uint32)
            self.assertEqual(result['statistics']['completed_samples'],128)
            self.assertEqual(model.sample(0,0)['counts'].shape,(0,2))

    def test_weak_tap_coherent_poisson_limit(self):
        # epsilon=1e-6 gives negligible finite-tap bias at lambda=0.5;
        # this checks photon statistics, rather than mirroring the native code.
        with self.model(v=np.ones((1,1),complex),x=np.zeros((2,2)),mean=[1,0],
                        tap_strength=1e-6,workers=16) as model:
            counts=model.sample(100000,716002)['counts'][:,0]
        hist=np.bincount(counts,minlength=12);expected=np.array([np.exp(-.5)*.5**k/math.factorial(k) for k in range(len(hist))])
        self.assertLess(abs(hist/len(counts)-expected).sum()/2,.01)
        self.assertGreater(counts.max(),1)

    def test_schedule_replay_with_displacement(self):
        root=Path(__file__).resolve().parents[1];data=np.load(root/'results/iteration71_pnr_branch_pilot_v2/input_3.npz')
        with self.model(v=data['v'],x=data['x'],noise_variance=data['variance']) as model:
            a=model.sample(2001,716003,workers=1);b=model.sample(2001,716003,workers=16)
            np.testing.assert_array_equal(a['counts'],b['counts'])
            self.assertEqual(b['statistics']['completed_samples'],2001)
            self.assertEqual(b['statistics']['photons'],b['counts'].sum())

    def test_public_interface_replays_raw_holdout(self):
        root=Path(__file__).resolve().parents[1];data=np.load(root/'results/iteration71_pnr_branch_pilot_v2/input_3.npz')
        expected=np.load(root/'results/iteration71_pnr_holdout/counts_3_branches_0.npz')['counts']
        with self.model(v=data['v'],x=data['x'],noise_variance=data['variance'],workers=16) as model:
            actual=model.sample(1024,0x74000000000000)['counts']
        np.testing.assert_array_equal(actual,expected[:1024])

    def test_invalid_parameters(self):
        for kwargs in ({'branch_budget':0},{'tap_strength':0},{'tap_strength':1},
                       {'tap_strength':np.nan},{'tap_strength':True},{'condition_trigger':.5}):
            with self.assertRaises(ValueError):self.model(**kwargs)
        with self.model() as model:
            for count,seed in ((-1,0),(True,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_native_failure_and_recovery(self):
        with self.model() as model:
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(10,0)
            model.mean[0]=0
            self.assertFalse(model.sample(10,0)['counts'].any())

if __name__=='__main__':unittest.main()
