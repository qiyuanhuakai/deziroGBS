import os
from pathlib import Path
import shutil
import unittest
import math
import numpy as np
from gbslab.native_pnr_newton import NativeNewtonPNRSampler

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('GBSLAB_NEWTON_PNR_LIBRARY'),'explicit Newton PNR library required')
class NativeNewtonPNRTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeNewtonPNRSampler(kwargs.pop('v',np.eye(2,dtype=complex)),
            kwargs.pop('x',np.zeros((4,4))),os.environ['GBSLAB_NEWTON_PNR_LIBRARY'],
            runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)

    def test_vacuum_and_empty(self):
        with self.model(workers=8) as model:
            r=model.sample(128,72003)
            self.assertFalse(r['counts'].any())
            self.assertEqual(r['statistics']['root_iterations'],0)
            self.assertEqual(r['statistics']['completed_samples'],128)
            self.assertEqual(model.sample(0,0)['counts'].shape,(0,2))

    def test_coherent_poisson_limit(self):
        with self.model(v=np.ones((1,1),complex),x=np.zeros((2,2)),mean=[1,0],tap_strength=1e-6,workers=16) as model:
            r=model.sample(100000,72004);counts=r['counts'][:,0]
        hist=np.bincount(counts,minlength=12)
        target=np.array([np.exp(-.5)*.5**k/math.factorial(k) for k in range(len(hist))])
        self.assertLess(abs(hist/len(counts)-target).sum()/2,.01)
        self.assertGreater(counts.max(),1)
        self.assertLess(r['statistics']['maximum_survival_residual'],1e-10)

    def test_old_holdout_and_worker_replay(self):
        data=np.load(ROOT/'results/iteration71_pnr_branch_pilot_v2/input_3.npz')
        expected=np.load(ROOT/'results/iteration71_pnr_holdout/counts_3_branches_0.npz')['counts'][:1024]
        with self.model(v=data['v'],x=data['x'],noise_variance=data['variance']) as model:
            a=model.sample(1024,0x74000000000000,workers=1)
            b=model.sample(1024,0x74000000000000,workers=16)
        np.testing.assert_array_equal(a['counts'],expected)
        np.testing.assert_array_equal(a['counts'],b['counts'])
        self.assertEqual(b['statistics']['photons'],b['counts'].sum())
        self.assertEqual(b['statistics']['bisection_fallback_roots'],0)

    def test_validation_failure_recovery_and_closed(self):
        for kw in ({'tap_strength':0},{'tap_strength':1},{'branch_budget':0},{'condition_trigger':.5}):
            with self.assertRaises(ValueError):self.model(**kw)
        with self.model() as model:
            for count,seed in ((-1,0),(True,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(10,0)
            model.mean[0]=0
            self.assertFalse(model.sample(10,0)['counts'].any())
        with self.assertRaises(RuntimeError):model.sample(1,0)

if __name__=='__main__':unittest.main()
