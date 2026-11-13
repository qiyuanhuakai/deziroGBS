import os,shutil,unittest
from pathlib import Path
import numpy as np
from gbslab.native_joint_number_pnr import NativeJointNumberPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_JOINT_PNR_LIBRARY'),'explicit joint library required')
class JointPNRTests(unittest.TestCase):
    def model(self,**kw):return NativeJointNumberPNRSampler(kw.pop('v',np.eye(2,dtype=complex)),kw.pop('t',.4),os.environ['GBSLAB_JOINT_PNR_LIBRARY'],runtime_dirs=[Path(shutil.which('g++')).parent],**kw)
    def test_exact_branch_replays_archive(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz')
        with self.model(v=d['v'],t=d['t'],exact_total_budget=64,joint_block_size=4) as model:r=model.sample(1024,0x88000000000000+2**40)
        np.testing.assert_array_equal(r['counts'],np.load(ROOT/'results/iteration79_pilot_v1/strict_replay.npz')['counts'][:1024])
    def test_phase_sensitive_pairs_remain_joint(self):
        v=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        with self.model(v=v,exact_total_budget=0,joint_block_size=2) as model:r=model.sample(1024,791001,fixed_total=14)
        np.testing.assert_array_equal(r['counts'],np.full((1024,2),7));self.assertEqual(r['statistics']['joint_blocks'],1024*7)
    def test_block_partition_and_worker_invariance(self):
        for b in (2,4,6,8,12):
            with self.model(exact_total_budget=0,joint_block_size=b) as model:a=model.sample(257,791002,fixed_total=14,workers=1);c=model.sample(257,791002,fixed_total=14,workers=16)
            np.testing.assert_array_equal(a['counts'],c['counts']);self.assertTrue(np.all(a['counts'].sum(axis=1)==14));self.assertFalse((a['counts']%2).any());self.assertEqual(a['statistics']['joint_blocks'],257*((14+b-1)//b))
    def test_vacuum_and_empty(self):
        with self.model(t=0.,exact_total_budget=0) as model:self.assertFalse(model.sample(100,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,2))
    def test_invalid_block_and_odd_total(self):
        for b in (True,0,1,3,14,2.0):
            with self.assertRaises(ValueError):self.model(joint_block_size=b)
        with self.model() as model:
            with self.assertRaises(ValueError):model.sample(1,0,fixed_total=3)
    def test_large_output_on_demand_pair_path(self):
        v=np.full((1025,1),1/np.sqrt(1025),complex)
        with self.model(v=v,exact_total_budget=0) as model:r=model.sample(64,791003,fixed_total=4)
        self.assertTrue(np.all(r['counts'].sum(axis=1)==4));self.assertGreater(np.count_nonzero(r['counts'].sum(axis=0)),100)
