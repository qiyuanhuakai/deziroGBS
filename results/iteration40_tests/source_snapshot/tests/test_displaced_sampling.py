import itertools
import math
import unittest
import numpy as np
from gbslab.displaced_sampling import DisplacedThresholdSampler,coherent_direction,refined_amplitudes,select_positive,prior_displacements
from gbslab.displaced_marginals import displacement_vacuum_effect,vacuum_expectation
from gbslab.mps_probe import dense_to_mps


def dense_threshold_law(dense,beta):
    """Independent Kronecker contraction of the binary finite-input POVMs."""
    effects=[]
    for d,b in zip(dense.shape,beta):
        v=np.array([np.exp(-abs(b)**2/2)*b**n/math.sqrt(math.factorial(n)) for n in range(d)])
        E=np.outer(v,v.conj()); effects.append((E,np.eye(d)-E))
    flat=dense.ravel(); law=[]
    for bits in itertools.product((0,1),repeat=dense.ndim):
        operator=np.ones((1,1),complex)
        for E,bit in zip(effects,bits): operator=np.kron(operator,E[bit])
        value=np.vdot(flat,operator@flat)
        if abs(value.imag)>1e-12 or value.real<0: raise ArithmeticError('Invalid dense threshold reference')
        law.append(value.real)
    return np.array(law)


def enumerate_refined(sampler,beta):
    """Enumerate all hidden trajectories; no Monte Carlo acceptance threshold."""
    law=np.zeros(2**sampler.modes); count=0
    def visit(site,env,probability,bits):
        nonlocal count
        if site==sampler.modes:
            law[bits]+=probability; count+=1; return
        A=sampler.tensors[site]; M=np.stack([env@A[:,n,:] for n in range(A.shape[1])],axis=1)
        transformed,masses,weights=refined_amplitudes(M,np.array([beta[site]])); total=math.fsum(weights[0])
        for label,p in enumerate(weights[0]):
            if p==0: continue
            physical=0 if label<=1 else label-1
            visit(site+1,transformed[:,physical,:]/math.sqrt(masses[0,physical]),probability*p/total,2*bits+(label!=0))
    visit(0,np.ones((1,1),complex),1.,0)
    return law,count


def dense_partial_trace(dense,sites):
    sites=tuple(sites); remaining=tuple(i for i in range(dense.ndim) if i not in sites)
    matrix=dense.transpose(sites+remaining).reshape(math.prod(dense.shape[i] for i in sites),-1)
    return matrix@matrix.conj().T


def mixed_dense_law(dense,F):
    """All vacuum subsets, followed by inclusion/exclusion; no sampling routines."""
    m=dense.ndim; subset={0:1.}
    for mask in range(1,2**m):
        sites=[i for i in range(m) if mask&(1<<i)]; dims=tuple(dense.shape[i] for i in sites)
        E=displacement_vacuum_effect(F[sites+[i+m for i in sites]],dims)
        subset[mask]=vacuum_expectation(dense_partial_trace(dense,sites),E)
    result=[]
    for bits in itertools.product((0,1),repeat=m):
        vacuum=sum(1<<i for i,b in enumerate(bits) if not b); clicks=[i for i,b in enumerate(bits) if b]
        terms=[]
        for include in itertools.product((0,1),repeat=len(clicks)):
            mask=vacuum+sum((1<<i) for i,take in zip(clicks,include) if take)
            terms.append((-1)**sum(include)*subset[mask])
        result.append(math.fsum(terms))
    result=np.array(result)
    if np.any(result<0) or abs(result.sum()-1)>1e-12: raise ArithmeticError('Invalid mixed reference')
    return result


class DisplacedSamplingTests(unittest.TestCase):
    def state(self,shape,seed):
        rng=np.random.default_rng(seed); dense=rng.normal(size=shape)+1j*rng.normal(size=shape); dense/=np.linalg.norm(dense)
        tensors,_=dense_to_mps(dense,128); return dense,tensors

    def test_positive_refinement_reconstructs_binary_effects(self):
        for d,b in itertools.product((1,2,4,8),(0j,.01+.02j,.6-.4j,2+3j)):
            c,w,tail=coherent_direction(np.array([b]),d)
            transformed,_,_=refined_amplitudes(np.eye(d,dtype=complex)[None],np.array([b]))
            E0=w[0]*np.outer(transformed[0,0].conj(),transformed[0,0])
            E1=tail[0]*np.outer(transformed[0,0].conj(),transformed[0,0])+transformed[0,1:].conj().T@transformed[0,1:]
            v=np.array([np.exp(-abs(b)**2/2)*b**n/math.sqrt(math.factorial(n)) for n in range(d)])
            np.testing.assert_allclose(E0,np.outer(v,v.conj()),atol=4e-15,rtol=3e-14)
            np.testing.assert_allclose(E0+E1,np.eye(d),atol=2e-15,rtol=0)

    def test_hidden_trajectory_sum_matches_entire_dense_law(self):
        for shape in ((2,3),(3,2,3),(4,4,4)):
            dense,tensors=self.state(shape,301001); beta=np.linspace(.2,.6,len(shape))*(1+.7j)
            law,count=enumerate_refined(DisplacedThresholdSampler(tensors),beta)
            self.assertEqual(count,math.prod(d+1 for d in shape))
            np.testing.assert_allclose(law,dense_threshold_law(dense,beta),atol=8e-15,rtol=0)

    def test_zero_displacement_fock_products_and_endpoint_uniforms(self):
        tensors=[]; occupations=(0,2,1,0)
        for n in occupations:
            a=np.zeros((1,3,1),complex); a[0,n,0]=1; tensors.append(a)
        sampler=DisplacedThresholdSampler(tensors); u=np.array([[0]*4,[np.nextafter(1.,0)]*4])
        out=sampler.sample(np.zeros((2,4)),u)
        np.testing.assert_array_equal(out['clicks'],np.array([[0,1,1,0]]*2))

    def test_tiny_positive_click_tail_is_not_subtracted_from_one(self):
        c,w,tail=coherent_direction(np.array([1e-8]),1)
        self.assertGreater(tail[0],0); self.assertAlmostEqual(tail[0]/1e-16,1,delta=1e-14)
        result=DisplacedThresholdSampler([np.ones((1,1,1))]).sample(np.array([[1e-8],[1e-8]]),np.array([[0],[.5]]))
        np.testing.assert_array_equal(result['clicks'][:,0],[1,0])
        self.assertGreater(result['selected_refined_probabilities'][0,0],0)

    def test_underflow_and_invalid_inputs_reject(self):
        for beta,d in ((1e-200,1),(1e-50,8),(40,4)):
            with self.assertRaises(ArithmeticError): coherent_direction(np.array([beta]),d)
        with self.assertRaises(ValueError): select_positive([[1]],np.array([1.]))
        with self.assertRaises(ValueError): prior_displacements(np.ones((3,2)),np.ones((1,2)))
        with self.assertRaises(ValueError): prior_displacements(np.ones((2,1),complex),np.ones((1,1)))

    def test_categorical_zero_masses_and_order(self):
        labels,total=select_positive(np.array([[0,.8,1e-25,.2]]*3),np.array([0,.1,.9]))
        np.testing.assert_array_equal(labels,[2,3,1]); np.testing.assert_allclose(total,1,atol=0,rtol=0)

    def test_empty_batch_and_batch_partition_reproducibility(self):
        _,tensors=self.state((3,3,3),301002); sampler=DisplacedThresholdSampler(tensors); rng=np.random.default_rng(301003)
        beta=.5*(rng.normal(size=(31,3))+1j*rng.normal(size=(31,3))); u=rng.random((31,3)); whole=sampler.sample(beta,u)
        split=[sampler.sample(beta[:11],u[:11]),sampler.sample(beta[11:],u[11:])]
        np.testing.assert_array_equal(whole['refined_labels'],np.concatenate([x['refined_labels'] for x in split]))
        self.assertEqual(sampler.sample(np.empty((0,3)),np.empty((0,3)))['clicks'].shape,(0,3))

    def test_complete_conditional_empirical_law(self):
        dense,tensors=self.state((3,3,3),301004); beta=np.array([.1+.2j,.4-.3j,.7+.2j]); p=dense_threshold_law(dense,beta)
        n=32768; rng=np.random.default_rng(301005); out=DisplacedThresholdSampler(tensors).sample(np.tile(beta,(n,1)),rng.random((n,3)))
        ids=out['clicks']@np.array([4,2,1]); empirical=np.bincount(ids,minlength=8)/n
        self.assertTrue(np.all(abs(empirical-p)<6*np.sqrt(p*(1-p)/n)+1/n))

    def test_mixed_vacuum_analytic_thermal_output_law(self):
        n=32768; sigma=np.array([.5,1.,1.5]); F=np.diag(np.r_[sigma,sigma]); rng=np.random.default_rng(301006)
        beta=prior_displacements(F,rng.normal(size=(n,6))); out=DisplacedThresholdSampler([np.ones((1,1,1))]*3).sample(beta,rng.random((n,3)))
        q=1/(1+sigma**2/2); p=np.array([math.prod((1-q[i]) if b else q[i] for i,b in enumerate(bits)) for bits in itertools.product((0,1),repeat=3)])
        empirical=np.bincount(out['clicks']@np.array([4,2,1]),minlength=8)/n
        self.assertTrue(np.all(abs(empirical-p)<6*np.sqrt(p*(1-p)/n)+1/n))

    def test_mixed_entangled_state_against_integrated_full_law(self):
        dense,tensors=self.state((3,3,3),301007); rng=np.random.default_rng(301008); F=.35*rng.normal(size=(6,3)); p=mixed_dense_law(dense,F)
        n=32768; beta=prior_displacements(F,rng.normal(size=(n,3))); out=DisplacedThresholdSampler(tensors).sample(beta,rng.random((n,3)))
        empirical=np.bincount(out['clicks']@np.array([4,2,1]),minlength=8)/n
        self.assertTrue(np.all(abs(empirical-p)<6*np.sqrt(p*(1-p)/n)+1/n))

    def test_prior_noise_mapping_not_vacuum_tilt(self):
        F=np.array([[1.,2.],[-3.,4.],[5.,6.],[7.,-8.]]); z=np.array([[.2,-.1],[1.,2.]])
        beta=prior_displacements(F,z); mu=z@F.T
        np.testing.assert_array_equal(beta,-(mu[:,:2]+1j*mu[:,2:])/2)

    def test_global_scaling_does_not_change_output_law(self):
        dense,tensors=self.state((3,3),301009); tensors[0]*=(2+3j); sampler=DisplacedThresholdSampler(tensors)
        law,_=enumerate_refined(sampler,np.array([.2j,.5])); np.testing.assert_allclose(law,dense_threshold_law(dense,[.2j,.5]),atol=3e-15,rtol=0)


if __name__=='__main__': unittest.main()
