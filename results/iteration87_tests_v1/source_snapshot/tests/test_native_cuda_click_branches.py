import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from gbslab.native_cuda_click_branches import NativeCudaBranchClickSampler

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('GBSLAB_CUDA_BRANCH_LIBRARY') and os.environ.get('GBSLAB_CLICK_REPLAY_LIBRARY'),'explicit CUDA and CPU replay libraries required')
class NativeCudaBranchTests(unittest.TestCase):
    def model(self,**kwargs):
        return NativeCudaBranchClickSampler(kwargs.pop('v',np.eye(2,dtype=complex)),
            kwargs.pop('x',np.zeros((4,4))),os.environ['GBSLAB_CLICK_REPLAY_LIBRARY'],
            gpu_library=os.environ['GBSLAB_CUDA_BRANCH_LIBRARY'],
            runtime_dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent],**kwargs)

    def test_vacuum_empty_and_closed(self):
        with self.model() as model:
            result=model.sample(128,72001)
            self.assertFalse(result['clicks'].any())
            self.assertFalse(result['conditional_click_probabilities'].any())
            self.assertEqual(result['statistics']['completed_samples'],128)
            self.assertEqual(model.sample(0,0)['clicks'].shape,(0,2))
        with self.assertRaises(RuntimeError):model.sample(1,0)

    def test_coherent_joint_law(self):
        w=np.array([1.,.7,-.2,.4]);p=-np.expm1(-(w[:2]**2+w[2:]**2)/2)
        patterns=np.array([[0,0],[1,0],[0,1],[1,1]],np.uint8)
        with self.model(mean=w) as model:
            r=model.replay(np.tile(w,(4,1)),np.where(patterns,0.,np.nextafter(1.,0.)))
        np.testing.assert_array_equal(r['clicks'],patterns)
        np.testing.assert_allclose(r['conditional_click_probabilities'],np.tile(p,(4,1)),atol=1e-13,rtol=0)

    def test_exact_small_law_and_budget_improvement(self):
        data=np.load(ROOT/'results/iteration71_branch_pilot_v1/input_3.npz')
        bits=np.array([[(mask>>i)&1 for i in range(4)] for mask in range(16)])
        with self.model(v=data['v'],x=data['x']) as model:
            errors=[]
            for budget in (1,4,8):
                r=model.replay(np.tile(data['mean'],(16,1)),np.where(bits,0.,np.nextafter(1.,0.)),branch_budget=budget)
                np.testing.assert_array_equal(r['clicks'],bits)
                law=np.prod(np.where(bits,r['conditional_click_probabilities'],1-r['conditional_click_probabilities']),axis=1)
                errors.append(abs(law-data['target']).sum()/2)
            self.assertLess(errors[-1],1e-12)
            self.assertLess(errors[1],errors[0]/100)

    def test_noisy_archived_record_replay(self):
        data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz')
        with self.model(v=data['v'],x=data['x0'],mean=data['mean'],noise_variance=data['variance']) as model:
            r=model.sample(32,715901)
        expected=np.load(ROOT/'results/iteration71_frugal_comparison/large_0_branch_4.npz')['clicks'][:32]
        np.testing.assert_array_equal(r['clicks'],expected)

    def test_model_inputs_are_owned(self):
        v=np.eye(2,dtype=complex);x=np.zeros((4,4))
        with self.model(v=v,x=x,mean=[1,0,0,0]) as model:
            before=model.sample(64,72002)['clicks'];v[:]=np.nan;x[:]=np.nan
            np.testing.assert_array_equal(before,model.sample(64,72002)['clicks'])

    def test_invalid_inputs_and_failure_recovery(self):
        for budget in (0,129,True,1.5):
            with self.assertRaises(ValueError):self.model(branch_budget=budget)
        with self.assertRaises(ValueError):self.model(v=np.ones((2,2),complex))
        with self.model() as model:
            for count,seed in ((-1,0),(True,0),(1048577,0),(1,-1),(1,2**64)):
                with self.assertRaises(ValueError):model.sample(count,seed)
            for draws in (np.ones((1,2)),np.full((1,2),np.nan),np.zeros((1,1))):
                with self.assertRaises(ValueError):model.replay(np.zeros((1,4)),draws)
            model.mean[0]=np.nan
            with self.assertRaises(ArithmeticError):model.sample(1,0)
            model.mean[0]=0
            # Finite but enormous coherent displacement underflows p0: fail explicitly.
            with self.assertRaises(ArithmeticError):model.replay(np.full((1,4),1000.),np.zeros((1,2)))
            self.assertFalse(model.sample(10,0)['clicks'].any())

if __name__=='__main__':unittest.main()
