import os,shutil,unittest,json
from pathlib import Path
import numpy as np
from gbslab.native_sphere_number_pnr import NativeSphereNumberPNRSampler
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_SPHERE_PNR_LIBRARY'),'explicit sphere library required')
class SpherePNRTests(unittest.TestCase):
    def model(self,**kwargs):return NativeSphereNumberPNRSampler(kwargs.pop('v',np.eye(2,dtype=complex)),kwargs.pop('t',.4),os.environ['GBSLAB_SPHERE_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kwargs)
    def test_exact_branch_archive_and_workers(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');old=np.load(ROOT/'results/iteration78_pilot_v2/strict_replay.npz')['counts'][:1024]
        with self.model(v=d['v'],t=d['t'],exact_total_budget=64) as model:
            a=model.sample(1024,0x88000000000000+2**40,workers=1);b=model.sample(1024,0x88000000000000+2**40,workers=16)
        np.testing.assert_array_equal(a['counts'],old);np.testing.assert_array_equal(a['counts'],b['counts'])
    def test_conditional_thermal_uniform_compositions(self):
        for v in (np.eye(2,dtype=complex),np.array([[1,1j],[1j,1]])/np.sqrt(2)):
            with self.model(v=v,exact_total_budget=0,workers=16) as model:r=model.sample(30000,781001,fixed_total=6)
            self.assertTrue(np.all(r['counts'].sum(axis=1)==6));hist=np.bincount(r['counts'][:,0],minlength=7)/30000;self.assertLess(abs(hist-1/7).max(),.012)
    def test_phase_sensitive_exact_and_approximate_branches(self):
        v=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        with self.model(v=v,exact_total_budget=2) as model:exact=model.sample(4096,781002,fixed_total=2)['counts']
        with self.model(v=v,exact_total_budget=0) as model:tail=model.sample(4096,781002,fixed_total=2)['counts']
        np.testing.assert_array_equal(exact,np.ones((4096,2),np.uint32));self.assertGreater(np.count_nonzero(tail[:,0]==0),1000)
    def test_single_input_tail_degenerates_to_multinomial(self):
        v=np.sqrt([.2,.3,.5]).astype(complex)[:,None];runtime=[Path(shutil.which('g++')).parent]
        with self.model(v=v,exact_total_budget=0,workers=16) as model:a=model.sample(4096,781003,fixed_total=12)
        with NativeBudgetNumberPNRSampler(v,.4,ROOT/'results/iteration77_budget_pilot_v1/budget.dll',exact_total_budget=0,workers=16,runtime_dirs=runtime) as model:b=model.sample(4096,781003,fixed_total=12)
        np.testing.assert_array_equal(a['counts'],b['counts'])
    def test_high_input_total_and_worker_replay(self):
        d=np.load(ROOT/'results/iteration77_budget_pilot_v1/input_4.npz')
        with self.model(v=d['v'],t=float(d['t']),exact_total_budget=12) as model:a=model.sample(256,781004,workers=1);b=model.sample(256,781004,workers=16)
        np.testing.assert_array_equal(a['counts'],b['counts']);np.testing.assert_array_equal(a['counts'].sum(axis=1),a['total_photons']);self.assertFalse((a['total_photons']%2).any());self.assertGreater(a['statistics']['tail_records'],240)
    def test_vacuum_and_resource_rejection(self):
        with self.model(t=0.,exact_total_budget=0) as model:self.assertFalse(model.sample(64,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,2))
        with self.model(v=np.eye(32,dtype=complex),exact_total_budget=22) as model:
            with self.assertRaisesRegex(ArithmeticError,'resource estimate'):model.sample(1,0)
