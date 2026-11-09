import math
from functools import lru_cache
import unittest
import numpy as np
from scipy.linalg import expm, logm

from gbslab.gaussian import haar_unitary, local_unitary
from gbslab.local_mps import (local_circuit, squeezed_product, passive_gate_blocks,
    LocalMPS, build_local_core, build_local_core_layerwise, sample_local_threshold, LocalSamplingError)
from gbslab.mps_probe import mps_to_dense, fixed_threshold_law


def gaussian_box(u, s, d):
    m=len(u); B=-math.tanh(s)*(u@u.T)
    @lru_cache(None)
    def coefficient(n):
        if sum(n)==0:
            return complex(math.cosh(s)**(-m/2))
        if sum(n)%2:
            return 0j
        i=next(j for j in range(m) if n[j])
        value=0j
        for j in range(m):
            remaining=n[j]-int(i==j)
            if remaining:
                prev=list(n); prev[i]-=1; prev[j]-=1
                value+=B[i,j]*math.sqrt(remaining)*coefficient(tuple(prev))
        return value/math.sqrt(n[i])
    x=np.empty((d,)*m,complex)
    for n in np.ndindex(x.shape):
        x[n]=coefficient(n)
    return x


class LocalMPSTests(unittest.TestCase):
    def test_circuit_matches_existing_ensemble(self):
        for m,depth in [(3,0),(4,4),(9,5)]:
            _,u=local_circuit(m,depth,np.random.default_rng(12001))
            np.testing.assert_allclose(u,local_unitary(m,depth,np.random.default_rng(12001)),atol=2e-15)

    def test_product_tail_and_exact_amplitudes(self):
        tensors,info=squeezed_product(3,.5,6)
        x=mps_to_dense(tensors)
        exact=gaussian_box(np.eye(3),.5,6)
        np.testing.assert_allclose(x,exact/np.linalg.norm(exact),atol=1e-15)
        self.assertAlmostEqual(1-np.linalg.norm(exact)**2,info['joint_input_tail'],delta=1e-14)

    def test_gate_against_full_sector_generator(self):
        u=haar_unitary(2,np.random.default_rng(12002)); d=5
        kept,leak,_=passive_gate_blocks(u,d)
        k=logm(u)
        for n in range(2*d-1):
            g=np.zeros((n+1,n+1),complex)
            for a in range(n+1):
                g[a,a]=k[0,0]*a+k[1,1]*(n-a)
                if a<n: g[a+1,a]=k[0,1]*math.sqrt((a+1)*(n-a))
                if a>0: g[a-1,a]=k[1,0]*math.sqrt(a*(n-a+1))
            ref=expm(g)
            for a in range(max(0,n-d+1),min(d-1,n)+1):
                for b in range(max(0,n-d+1),min(d-1,n)+1):
                    self.assertAlmostEqual(abs(kept[b*d+n-b,a*d+n-a]-ref[b,a]),0,delta=1e-13)
        np.testing.assert_allclose(kept.conj().T@kept+leak.conj().T@leak,np.eye(d*d),atol=2e-14)

    def test_hong_ou_mandel_boundary_does_not_fake_unitarity(self):
        u=np.array([[1,1],[-1,1]])/math.sqrt(2)
        kept,leak,_=passive_gate_blocks(u,2)
        self.assertLess(np.linalg.norm(kept[:,3]),1e-14)
        self.assertAlmostEqual(np.linalg.norm(leak[:,3]),1.,delta=1e-14)
        one=np.array([0,1],complex).reshape(1,2,1)
        mps=LocalMPS([one,one],2)
        with self.assertRaisesRegex(ArithmeticError,'retains no'):
            mps.pair(0,u)

    def test_center_motion_preserves_complex_state(self):
        tensors,_=squeezed_product(4,.4,4)
        mps=LocalMPS(tensors,8)
        mps.pair(0,haar_unitary(2,np.random.default_rng(12003)))
        before=mps_to_dense(mps.tensors)
        for position in [3,0,2,1,0]:
            mps.move_center(position)
            np.testing.assert_allclose(mps_to_dense(mps.tensors),before,atol=2e-14)

    def test_direct_dense_projected_circuit(self):
        m,d=3,4
        gates,_=local_circuit(m,4,np.random.default_rng(12004))
        tensors,_=squeezed_product(m,.4,d)
        x=mps_to_dense(tensors)
        losses=[]
        for gate in gates:
            if gate[0]=='phases':
                for i,phase in enumerate(gate[1]):
                    shape=[1]*m; shape[i]=d
                    x*=np.exp(1j*phase*np.arange(d)).reshape(shape)
            else:
                _,i,u=gate; k,_,_=passive_gate_blocks(u,d)
                order=[i,i+1]+[j for j in range(m) if j not in (i,i+1)]
                y=(k@x.transpose(order).reshape(d*d,-1)).reshape((d,)*m).transpose(np.argsort(order))
                losses.append(1-np.linalg.norm(y)**2)
                x=y/np.linalg.norm(y)
        result,info=build_local_core(m,.4,d,16,gates)
        np.testing.assert_allclose(mps_to_dense(result),x,atol=4e-14)
        np.testing.assert_allclose([r['local_projection_tail'] for r in info['gates']],losses,atol=2e-14)

    def test_trace_bound_against_untruncated_gaussian(self):
        for m,d,bond in [(3,4,2),(4,4,4),(4,6,8)]:
            gates,u=local_circuit(m,4,np.random.default_rng(12005))
            a,info=build_local_core(m,.3,d,bond,gates)
            exact=gaussian_box(u,.3,d)
            overlap=np.vdot(exact.ravel(),mps_to_dense(a).ravel())
            distance=math.sqrt(max(0,1-abs(overlap)**2))
            self.assertLessEqual(distance,info['construction_trace_budget']+2e-12)

    def test_bond_truncation_recorded(self):
        gates,_=local_circuit(4,2,np.random.default_rng(12006))
        _,info=build_local_core(4,.6,6,1,gates)
        self.assertGreater(sum(r['bond_tail'] for r in info['gates']),.01)
        self.assertEqual(info['max_bond_actual'],1)

    def test_exact_vacuum_for_many_modes(self):
        gates,_=local_circuit(32,4,np.random.default_rng(12007))
        tensors,info=build_local_core(32,0,2,4,gates)
        self.assertEqual(info['construction_trace_budget'],0.)
        self.assertLess(sum(a.size for a in tensors),1000)

    def test_resource_refusal(self):
        tensors,_=squeezed_product(4,.4,4)
        mps=LocalMPS(tensors,8,max_pair_elements=1)
        with self.assertRaises(MemoryError):
            mps.pair(0,np.eye(2))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError): squeezed_product(4,-.5,4)
        with self.assertRaises(ValueError): passive_gate_blocks(np.ones((2,2)),4)
        with self.assertRaises(ValueError): local_circuit(4,-1,np.random.default_rng(1))

    def test_sampling_matches_fixed_law(self):
        gates,_=local_circuit(3,3,np.random.default_rng(12008))
        tensors,_=build_local_core(3,.35,4,8,gates)
        alpha=np.array([.2j,.1+.3j,-.2j])
        law=fixed_threshold_law(tensors,alpha)
        samples,info=sample_local_threshold(tensors,np.tile(alpha,(10000,1)),np.random.default_rng(12009))
        empirical=np.bincount(samples@np.array([1,2,4]),minlength=8)/len(samples)
        self.assertLess(.5*np.abs(law-empirical).sum(),.02)
        self.assertEqual(info['completed_samples'],10000)

    def test_sampling_failure_retains_completed_samples(self):
        tensors,_=squeezed_product(2,.2,3)
        class FaultRNG:
            def __init__(self): self.calls=0
            def choice(self,*args,**kwargs):
                self.calls+=1
                if self.calls==4: raise ValueError('injected failure')
                return 0
        with self.assertRaises(LocalSamplingError) as cm:
            sample_local_threshold(tensors,np.zeros((4,2)),FaultRNG())
        self.assertEqual(cm.exception.completed_samples.shape,(1,2))
        self.assertEqual(cm.exception.sample_index,1)
        self.assertEqual(cm.exception.mode,1)
        self.assertEqual(cm.exception.prefix,[0])

    def test_rounding_joint_fidelity_not_sum_of_distances(self):
        tensors,_=squeezed_product(5,.5,4)
        mps=LocalMPS(tensors,2)
        gates,_=local_circuit(5,2,np.random.default_rng(12010))
        for gate in gates:
            if gate[0]=='phases': mps.phases(gate[1])
            else: mps.pair(gate[1],gate[2],truncate_bond=False)
        before=mps_to_dense(mps.tensors)
        losses,joint=mps.round_bonds()
        after=mps_to_dense(mps.tensors)
        actual=1-abs(np.vdot(before.ravel(),after.ravel()))**2
        self.assertAlmostEqual(joint,actual,delta=2e-14)
        self.assertGreater(joint,0)

    def test_disjoint_layer_projection_fidelity(self):
        tensors,_=squeezed_product(4,.5,4)
        mps=LocalMPS(tensors,16)
        u1=haar_unitary(2,np.random.default_rng(12011))
        u2=haar_unitary(2,np.random.default_rng(12012))
        before=mps_to_dense(tensors)
        k1,_,_=passive_gate_blocks(u1,4); k2,_,_=passive_gate_blocks(u2,4)
        unnormalized=(k1@before.reshape(16,16)@k2.T).reshape((4,)*4)
        expected_tail=1-np.linalg.norm(unnormalized)**2
        r1=mps.pair(0,u1,truncate_bond=False)
        r2=mps.pair(2,u2,truncate_bond=False)
        actual_tail=1-(1-r1['local_projection_tail'])*(1-r2['local_projection_tail'])
        self.assertAlmostEqual(expected_tail,actual_tail,delta=2e-14)
        np.testing.assert_allclose(mps_to_dense(mps.tensors),unnormalized/np.linalg.norm(unnormalized),atol=2e-14)

    def test_layerwise_bound_against_full_gaussian(self):
        for m,d,chi in [(4,4,2),(4,6,8),(5,4,4)]:
            gates,u=local_circuit(m,4,np.random.default_rng(12013))
            tensors,info=build_local_core_layerwise(m,.4,d,chi,gates)
            exact=gaussian_box(u,.4,d)
            actual=math.sqrt(max(0,1-abs(np.vdot(exact.ravel(),mps_to_dense(tensors).ravel()))**2))
            self.assertLessEqual(actual,info['construction_trace_budget']+1e-12)
            self.assertLessEqual(info['max_bond_actual'],chi)

    def test_layerwise_rejects_overlapping_gates(self):
        with self.assertRaisesRegex(ValueError,'disjoint'):
            build_local_core_layerwise(3,.3,4,4,[('pair',0,np.eye(2)),('pair',1,np.eye(2))])


if __name__=='__main__':
    unittest.main()
