import math
import sys
from pathlib import Path
import unittest
import numpy as np
from gbslab.adaptive_mps import schmidt_rank, adaptive_round_bonds, budgeted_attempt, build_adaptive_core, ConstructionBudgetUnmet
from gbslab.local_mps import LocalMPS, squeezed_product, local_circuit
from gbslab.mps_probe import mps_to_dense
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from iteration12_reference import gaussian_box_coefficients


class AdaptiveMPSTests(unittest.TestCase):
    def test_rank_minimality_and_cap_failure(self):
        s=np.sqrt([.8,.15,.04,.01])
        rank,tail,miss=schmidt_rank(s,.051,4)
        self.assertEqual(rank,2); self.assertAlmostEqual(tail,.05); self.assertFalse(miss)
        rank,tail,miss=schmidt_rank(s,.001,2)
        self.assertEqual(rank,2); self.assertAlmostEqual(tail,.05); self.assertTrue(miss)
        self.assertEqual(schmidt_rank([1,0,0],0,3),(1,0.,False))

    def test_tiny_tail_is_not_subtracted_from_one(self):
        rank,tail,miss=schmidt_rank([1,1e-12],1e-25,2)
        self.assertEqual(rank,2); self.assertFalse(miss)
        self.assertGreater(schmidt_rank([1,1e-12],0,1)[1],0)

    def test_adaptive_sweep_joint_fidelity(self):
        tensors,_=squeezed_product(4,.4,4); mps=LocalMPS(tensors,16)
        gates,_=local_circuit(4,3,np.random.default_rng(1501))
        for g in gates:
            if g[0]=='phases': mps.phases(g[1])
            else: mps.pair(g[1],g[2],truncate_bond=False)
        before=mps_to_dense(mps.tensors)
        info=adaptive_round_bonds(mps,.01); after=mps_to_dense(mps.tensors)
        self.assertAlmostEqual(1-abs(np.vdot(before.ravel(),after.ravel()))**2,info['joint_tail'],delta=2e-14)
        self.assertLessEqual(info['joint_tail'],.01+1e-14)
        self.assertEqual(info['rank_cap_misses'],0)

    def test_actual_gaussian_state_is_within_budget(self):
        for s in (.2,.4):
            gates,u=local_circuit(4,4,np.random.default_rng(1502))
            tensors,info=build_adaptive_core(4,s,gates,.1,dimensions=(4,6,8,10),caps=(4,8,16))
            exact=gaussian_box_coefficients(u,s,info['dimension'])
            distance=math.sqrt(max(0.,1-abs(np.vdot(exact.ravel(),mps_to_dense(tensors).ravel()))**2))
            self.assertLessEqual(distance,info['construction_trace_budget']+1e-12)
            self.assertLessEqual(info['construction_trace_budget'],.1)
            self.assertLessEqual(info['max_bond_actual'],info['cap'])
            self.assertEqual(info['attempts'][-1]['status'],'accepted')

    def test_input_rejection_retained(self):
        with self.assertRaises(ConstructionBudgetUnmet) as cm:
            build_adaptive_core(128,.8,[],.001,dimensions=(2,4),caps=(4,))
        self.assertEqual([r['status'] for r in cm.exception.attempts],['input_budget_rejected']*2)

    def test_resource_failure_not_success_or_missing(self):
        gates,_=local_circuit(4,2,np.random.default_rng(1503))
        with self.assertRaises(ConstructionBudgetUnmet) as cm:
            build_adaptive_core(4,.1,gates,.2,dimensions=(4,6),caps=(4,),max_pair_elements=1)
        self.assertEqual(len(cm.exception.attempts),1)
        self.assertEqual(cm.exception.attempts[0]['status'],'resource_failed')
        self.assertNotIn('construction_trace_budget',cm.exception.attempts[0])

    def test_zero_depth_and_callback(self):
        events=[]
        tensors,info=build_adaptive_core(3,0,[],.01,on_attempt=events.append)
        self.assertEqual(info['construction_trace_budget'],0)
        self.assertEqual(info['completed_layers'],0)
        self.assertEqual(events[0]['status'],'accepted')
        self.assertAlmostEqual(np.linalg.norm(mps_to_dense(tensors)),1)

    def test_invalid_policy_and_spectrum(self):
        for s in ([0,0],[1,-.1],[.1,1],[1,float('nan')]):
            with self.assertRaises(ValueError): schmidt_rank(s,.1,4)
        with self.assertRaises(ValueError): build_adaptive_core(4,.2,[],.1,dimensions=(6,4))
        with self.assertRaises(ValueError): budgeted_attempt(4,.2,4,4,[],1)


if __name__=='__main__': unittest.main()
