import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_philox_click import NativePhiloxClickSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_FUSED_CLICK_LIBRARY'),'explicit fused GPU library required')
class FusedClickTests(unittest.TestCase):
    def model(self,library,**kwargs):
        d=np.load(ROOT/'results/iteration71_branch_pilot_v1/input_4.npz')
        return NativePhiloxClickSampler(d['v'],d['x'],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=library,mean=d['mean'],noise_variance=d['variance'],runtime_dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent],**kwargs)
    def test_pair_replay_and_workspace(self):
        for budget in (1,3,4,128):
            with self.model(ROOT/'results/iteration73_philox_pilot_v1/philox.dll',branch_budget=budget) as old,self.model(os.environ['GBSLAB_FUSED_CLICK_LIBRARY'],branch_budget=budget) as new:
                a=old.sample(257,744301);b=new.sample(257,744301);d=2*new.inputs
            np.testing.assert_array_equal(a['clicks'],b['clicks']);np.testing.assert_allclose(a['conditional_click_probabilities'],b['conditional_click_probabilities'],atol=1e-9,rtol=0)
            self.assertEqual(a['cuda_timings']['workspace_bytes']-b['cuda_timings']['workspace_bytes'],257*budget*(d*d+d)*8)
    def test_batch_continuation(self):
        with self.model(os.environ['GBSLAB_FUSED_CLICK_LIBRARY'],branch_budget=3) as model:
            a=model.sample(257,744302,row_offset=11);b=model.sample(101,744302,row_offset=11);c=model.sample(156,744302,row_offset=112)
        np.testing.assert_array_equal(a['clicks'],np.concatenate([b['clicks'],c['clicks']]))
    def test_archived_large_replay(self):
        data=np.load(ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz')
        with NativePhiloxClickSampler(data['v'],data['x0'],ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll',gpu_library=os.environ['GBSLAB_FUSED_CLICK_LIBRARY'],mean=data['mean'],noise_variance=data['variance'],branch_budget=4,runtime_dirs=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent]) as model:r=model.sample(64,743201)
        old=np.load(ROOT/'results/iteration74_fused_comparison_v2/records_22_0_fused.npz');np.testing.assert_array_equal(r['clicks'],old['clicks'][:64])
