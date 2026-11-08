import math
import sys
from pathlib import Path
import unittest
import numpy as np
from gbslab.local_mps import LocalMPS, local_circuit, squeezed_product
from gbslab.mps_probe import mps_to_dense
from gbslab.checkpoint_mps import rolling_round_bonds, checkpoint_attempt, build_checkpoint_core, layer_schedule
from gbslab.adaptive_mps import ConstructionBudgetUnmet
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from iteration12_reference import gaussian_box_coefficients


class CheckpointMPSTests(unittest.TestCase):
    def test_rolling_joint_fidelity_and_allowances(self):
        tensors,_=squeezed_product(4,.4,4); mps=LocalMPS(tensors,16)
        gates,_=local_circuit(4,3,np.random.default_rng(1601))
        for g in gates:
            if g[0]=='phases': mps.phases(g[1])
            else: mps.pair(g[1],g[2],truncate_bond=False)
        before=mps_to_dense(mps.tensors); info=rolling_round_bonds(mps,.01)
        after=mps_to_dense(mps.tensors)
        self.assertAlmostEqual(info['joint_tail'],1-abs(np.vdot(before.ravel(),after.ravel()))**2,delta=3e-14)
        self.assertLessEqual(info['joint_tail'],.01+1e-14)
        spent=0.
        for i,(q,t) in enumerate(zip(info['conditional_losses'],info['per_cut_allowances'])):
            expected=-math.expm1(min(0,math.log1p(-.01)-spent)/(3-i))
            self.assertAlmostEqual(t,expected,delta=1e-17); spent+=math.log1p(-q)

    def test_zero_loss_releases_later_allowance(self):
        tensors,_=squeezed_product(4,0,4); mps=LocalMPS(tensors,8)
        info=rolling_round_bonds(mps,.01)
        self.assertEqual(info['joint_tail'],0)
        self.assertGreater(info['per_cut_allowances'][-1],info['per_cut_allowances'][0])

    def test_checkpoint_retries_and_original_gaussian_bound(self):
        gates,u=local_circuit(4,4,np.random.default_rng(1602))
        tensors,info=build_checkpoint_core(4,.4,gates,.1,dimensions=(6,8,10),caps=(1,2,4,8,16,32))
        exact=gaussian_box_coefficients(u,.4,info['dimension'])
        actual=mps_to_dense(tensors)
        distance=math.sqrt(max(0,1-abs(np.vdot(exact.ravel(),actual.ravel()))**2))
        self.assertLessEqual(distance,info['construction_trace_budget']+1e-12)
        self.assertLessEqual(info['construction_trace_budget'],.1)
        self.assertTrue(any(len(layer['rounding_attempts'])>1 for layer in info['layers']))
        # The final dimension attempt executes each optical gate once, despite rank retries.
        self.assertEqual(info['completed_gates'],sum(g[0]=='pair' for g in gates))

    def test_projection_stop_has_no_final_state_or_bound(self):
        gates,_=local_circuit(4,4,np.random.default_rng(1603))
        _,initial=squeezed_product(4,.4,4)
        tensors,info=checkpoint_attempt(4,.4,4,(8,16),gates,initial['input_trace_budget']+1e-6)
        self.assertIsNone(tensors); self.assertEqual(info['status'],'projection_budget_exhausted')
        self.assertNotIn('construction_trace_budget',info)

    def test_checkpoint_resource_stop_is_recorded(self):
        gates,_=local_circuit(4,2,np.random.default_rng(1604))
        with self.assertRaises(ConstructionBudgetUnmet) as cm:
            build_checkpoint_core(4,.2,gates,.1,dimensions=(6,),max_checkpoint_elements=1)
        self.assertEqual(cm.exception.attempts[0]['status'],'resource_failed')
        self.assertIn('Checkpoint copy cap',cm.exception.attempts[0]['error'])

    def test_rank_exhaustion_is_not_certified_output(self):
        gates,_=local_circuit(4,3,np.random.default_rng(1605))
        tensors,info=checkpoint_attempt(4,.3,6,(1,),gates,.1)
        self.assertIsNone(tensors); self.assertEqual(info['status'],'rank_budget_unmet')
        self.assertNotIn('construction_trace_budget',info)
        self.assertTrue(info['layers'][-1]['rounding_attempts'])

    def test_empty_circuit_is_normalized(self):
        tensors,info=build_checkpoint_core(3,0,[],.1,dimensions=(4,))
        self.assertEqual(info['construction_trace_budget'],0)
        self.assertAlmostEqual(np.linalg.norm(mps_to_dense(tensors)),1)

    def test_equal_allocation_control(self):
        gates,_=local_circuit(4,3,np.random.default_rng(1606))
        for mode in ('equal','rolling'):
            tensors,info=build_checkpoint_core(4,.3,gates,.1,dimensions=(6,8),allocation=mode)
            self.assertEqual(info['allocation'],mode)
            self.assertLessEqual(info['construction_trace_budget'],.1)

    def test_invalid_layers_and_policy(self):
        for gates in ([('pair',0,np.eye(2))],[('phases',[0,0,0]),('pair',0,np.eye(2)),('pair',1,np.eye(2))]):
            with self.assertRaises(ValueError): layer_schedule(3,gates)
        with self.assertRaises(ValueError): checkpoint_attempt(3,.2,4,(8,4),[],.1)
        with self.assertRaises(ValueError): rolling_round_bonds(None,1)


if __name__=='__main__': unittest.main()
