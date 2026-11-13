import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_direct_click import NativeDirectClickSampler

@unittest.skipUnless(os.environ.get('GBSLAB_BRANCH_CLICK_LIBRARY'),'explicit signed-branch library required')
class NativeBranchClickTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeBranchClickSampler(kwargs.pop('v',np.eye(2,dtype=complex)),
            kwargs.pop('x',np.zeros((4,4))),os.environ['GBSLAB_BRANCH_CLICK_LIBRARY'],
            runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)

    def test_vacuum_and_impossible_prefix(self):
        with self.model() as model:
            self.assertFalse(model.sample(100,71)['clicks'].any())
            self.assertEqual(model.prefix_probability([])['probability'],1)
            self.assertEqual(model.prefix_probability([0,0])['probability'],1)
            self.assertEqual(model.prefix_probability([1,0])['probability'],0)
            self.assertEqual(model.sample(0,0)['clicks'].shape,(0,2))

    def test_correlated_exact_law_and_budget_improvement(self):
        root=Path(__file__).resolve().parents[1]
        inp=np.load(root/'results/iteration71_branch_pilot_v1/input_3.npz')
        with self.model(v=inp['v'],x=inp['x']) as model:
            laws=[]
            for budget in (1,4,8):
                law=np.array([model.prefix_probability([(mask>>j)&1 for j in range(4)],branch_budget=budget)['probability'] for mask in range(16)])
                self.assertAlmostEqual(law.sum(),1,places=12);laws.append(law)
            self.assertLess(abs(laws[2]-inp['target']).sum()/2,1e-12)
            self.assertLess(abs(laws[1]-inp['target']).sum(),abs(laws[0]-inp['target']).sum()/100)

    def test_coherent_joint_law(self):
        mean=np.array([1.,.7,-.2,.4]);p=-np.expm1(-(mean[:2]**2+mean[2:]**2)/2)
        with self.model(mean=mean) as model:
            for a in (0,1):
                for b in (0,1):
                    expected=(p[0] if a else 1-p[0])*(p[1] if b else 1-p[1])
                    self.assertAlmostEqual(model.prefix_probability([a,b])['probability'],expected,places=13)

    def test_schedule_and_budget_one_replay(self):
        rng=np.random.default_rng(711712);v=np.linalg.qr(rng.normal(size=(8,3))+1j*rng.normal(size=(8,3)))[0]
        x=np.diag(np.r_[np.full(3,.4),np.full(3,-.15)]);noise=np.r_[np.full(3,.1),np.zeros(3)]
        with self.model(v=v,x=x,noise_variance=noise) as model:
            for budget in (1,2,8):
                a=model.sample(512,718001,workers=1,branch_budget=budget)
                b=model.sample(512,718001,workers=16,branch_budget=budget)
                np.testing.assert_array_equal(a['clicks'],b['clicks'])
                self.assertEqual(b['statistics']['completed_samples'],512)
            with NativeDirectClickSampler(v,x,os.environ['GBSLAB_BRANCH_CLICK_LIBRARY'],noise_variance=noise,runtime_dirs=[Path(shutil.which('g++')).parent]) as old:
                np.testing.assert_array_equal(old.sample(512,718001)['clicks'],model.sample(512,718001,branch_budget=1)['clicks'])

    def test_noise_prefix_requires_realization(self):
        with self.model(noise_variance=[.1,0,0,0]) as model:
            with self.assertRaises(ValueError):model.prefix_probability([0])
            self.assertEqual(model.prefix_probability([0],mean=np.zeros(4))['probability'],1)

    def test_invalid_inputs_and_closed_lifetime(self):
        for budget in (0,-1,65537,True,1.5):
            with self.assertRaises(ValueError):self.model(branch_budget=budget)
        with self.model() as model:
            for bits in ([2],[-1],[np.nan],[[0]],[0,0,0]):
                with self.assertRaises(ValueError):model.prefix_probability(bits)
            for count,seed in ((-1,0),(True,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
        with self.assertRaises(RuntimeError):model.prefix_probability([0])
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_native_failure_does_not_poison_next_call(self):
        with self.model(workers=4) as model:
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(20,0)
            model.mean[0]=0
            self.assertFalse(model.sample(20,0)['clicks'].any())

if __name__=='__main__':unittest.main()
