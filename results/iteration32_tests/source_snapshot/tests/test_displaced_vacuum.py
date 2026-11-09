import functools
import itertools
import math
import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from gbslab.displaced_vacuum import CoherentMPS,GaussianVacuumProposal,importance_summary
from gbslab.mps_probe import dense_to_mps
from gbslab.gaussian import haar_unitary
from gbslab.exact import ThresholdModel
from gbslab.displaced_marginals import local_reduced_states,displacement_vacuum_effect,vacuum_expectation,two_mode_threshold


def dense_coherent_log(dense,beta):
    beta=np.asarray(beta); amplitudes=np.zeros(len(beta),complex)
    for n in np.ndindex(dense.shape):
        factor=np.ones(len(beta),complex)
        for i,k in enumerate(n): factor*=beta[:,i].conj()**k/math.sqrt(math.factorial(k))
        amplitudes+=dense[n]*factor
    with np.errstate(divide='ignore'): return 2*np.log(abs(amplitudes))-np.sum(abs(beta)**2,axis=1)


def gaussian_moment_vacuum(dense,noise):
    """Independent dense Fock double sum using centered Gaussian Wick moments."""
    m=dense.ndim; L=np.asarray(noise,float); K=np.eye(L.shape[1])+L.T@L/2
    latent=np.linalg.inv(K); H=-(L[:m]+1j*L[m:])/2; linear=np.vstack([H.conj(),H]); covariance=linear@latent@linear.T
    @functools.lru_cache(None)
    def moment(powers):
        degree=sum(powers)
        if degree==0: return 1.+0j
        if degree%2: return 0j
        a=next(i for i,x in enumerate(powers) if x); rest=list(powers); rest[a]-=1; total=0j
        for j,count in enumerate(rest):
            if count:
                child=rest.copy(); child[j]-=1; total+=count*covariance[a,j]*moment(tuple(child))
        return total
    coefficients=[]
    for n in np.ndindex(dense.shape): coefficients.append((n,dense[n]/math.sqrt(math.prod(math.factorial(k) for k in n))))
    terms=[a*b.conjugate()*moment(n+k) for n,a in coefficients for k,b in coefficients]
    result=complex(math.fsum(t.real for t in terms),math.fsum(t.imag for t in terms))/math.sqrt(np.linalg.det(K))
    if abs(result.imag)>1e-12 or result.real<0: raise ArithmeticError('Invalid Wick reference')
    return result.real


def quadrature_vacuum(dense,noise,order=None):
    """Vacuum-reference tilt makes the finite Fock integrand a polynomial."""
    m=dense.ndim; order=order or sum(d-1 for d in dense.shape)+1; r=noise.shape[1]
    nodes,weights=hermgauss(order); grid=np.array(list(itertools.product(range(order),repeat=r)),int)
    standard=np.sqrt(2)*nodes[grid]; mass=np.prod(weights[grid]/math.sqrt(np.pi),axis=1)
    proposal=GaussianVacuumProposal(np.zeros((m,m)),noise); _,beta=proposal.from_standard(standard)
    ratios,_=proposal.likelihood_ratios(dense_coherent_log(dense,beta),beta)
    return proposal.mixed_vacuum*math.fsum(mass*ratios)


class DisplacedVacuumTests(unittest.TestCase):
    def test_coherent_batch_against_dense_complex_state_and_gauge(self):
        rng=np.random.default_rng(291001); dense=rng.normal(size=(4,3,3))+1j*rng.normal(size=(4,3,3)); dense/=np.linalg.norm(dense)
        tensors,_=dense_to_mps(dense,20); tensors[0]*=(2+3j); evaluator=CoherentMPS(tensors)
        beta=rng.normal(size=(17,3))+1j*rng.normal(size=(17,3))
        np.testing.assert_allclose(evaluator.log_probability(beta),dense_coherent_log(dense,beta),atol=3e-13,rtol=0)
        self.assertAlmostEqual(2*evaluator.input_log_norm,math.log(13),delta=1e-13)

    def test_reference_normalization_and_change_of_measure_identity(self):
        rng=np.random.default_rng(291002); U=haar_unitary(3,rng); B=-(U*np.tanh([.1,.3,.5]))@U.T; L=rng.normal(size=(6,4))*.4
        p=GaussianVacuumProposal(B,L); z,beta=p.from_standard(rng.normal(size=(37,4)))
        expected=ThresholdModel(p.covariance+L@L.T).vacuum_probability(tuple(range(3)))
        self.assertAlmostEqual(p.mixed_vacuum,expected,delta=2e-15)
        mu=z@L.T; log_cov=p.log_pure_vacuum-.5*np.einsum('bi,ij,bj->b',mu,np.linalg.inv(p.covariance+np.eye(6)),mu)
        np.testing.assert_allclose(p.gaussian_log_probability(beta),log_cov,atol=2e-14,rtol=0)
        ratio,_=p.likelihood_ratios(log_cov,beta); np.testing.assert_allclose(ratio,1,atol=2e-14,rtol=0)
        lhs=p.log_mixed_vacuum-p.gaussian_log_probability(beta)
        rhs=.5*np.einsum('bi,ij,bj->b',z,p.precision-np.eye(4),z)-.5*p.log_determinant
        np.testing.assert_allclose(lhs,rhs,atol=2e-14,rtol=0)

    def test_single_fock_analytic_thermal_noise(self):
        for n in range(4):
            dense=np.zeros(4,complex); dense[n]=1; noise=np.sqrt(.6)*np.eye(2); expected=.3**n/(1.3**(n+1))
            self.assertAlmostEqual(gaussian_moment_vacuum(dense,noise),expected,delta=2e-15)
            self.assertAlmostEqual(quadrature_vacuum(dense,noise),expected,delta=2e-15)

    def test_independent_wick_and_gauss_hermite(self):
        rng=np.random.default_rng(291003)
        for shape,r in [((4,4),2),((3,3,3),4)]:
            dense=rng.normal(size=shape)+1j*rng.normal(size=shape); dense/=np.linalg.norm(dense); noise=rng.normal(size=(2*len(shape),r))*.35
            self.assertAlmostEqual(gaussian_moment_vacuum(dense,noise),quadrature_vacuum(dense,noise),delta=3e-14)

    def test_importance_samples_against_analytic_fock_probability(self):
        p=GaussianVacuumProposal(np.zeros((1,1)),np.sqrt(.6)*np.eye(2)); state=np.array([0,1,0,0],complex).reshape(1,4,1); evaluator=CoherentMPS([state])
        _,beta=p.from_standard(np.random.default_rng(291004).normal(size=(12000,2))); w,_=p.likelihood_ratios(evaluator.log_probability(beta),beta); r=importance_summary(w,p.mixed_vacuum)
        self.assertLess(abs(r['probability_estimate']-.3/1.3**2),6*r['probability_standard_error'])
        self.assertGreater(r['empirical_effective_sample_size'],4000)

    def test_noiseless_vacuum_and_exact_zero_numeric_tracking(self):
        p=GaussianVacuumProposal(np.zeros((1,1)),np.zeros((2,0))); _,beta=p.from_standard(np.zeros((8,0)))
        e=CoherentMPS([np.array([1,0],complex).reshape(1,2,1)]); w,_=p.likelihood_ratios(e.log_probability(beta),beta)
        self.assertEqual(importance_summary(w,p.mixed_vacuum)['probability_estimate'],1)
        e=CoherentMPS([np.array([0,1],complex).reshape(1,2,1)]); w,_=p.likelihood_ratios(e.log_probability(beta),beta)
        self.assertEqual(importance_summary(w,p.mixed_vacuum)['numeric_zero_weights'],8)

    def test_positive_weight_underflow_not_dropped(self):
        p=GaussianVacuumProposal(np.zeros((1,1)),np.eye(2))
        with self.assertRaisesRegex(ArithmeticError,'no mass'): p.likelihood_ratios(np.array([-1000.]),np.zeros((1,1)))
        with self.assertRaisesRegex(ArithmeticError,'no mass'): p.likelihood_ratios(np.array([1000.]),np.zeros((1,1)))
        e=CoherentMPS([np.array([0,0,1],complex).reshape(1,3,1)])
        with self.assertRaisesRegex(ArithmeticError,'underflow'): e.log_probability(np.array([[1e-200]]))

    def test_invalid_data(self):
        with self.assertRaises(ValueError): GaussianVacuumProposal(np.zeros((1,1)),np.eye(2)*(1+1j))
        with self.assertRaises(ValueError): GaussianVacuumProposal(np.zeros((1,1)),np.eye(3))
        with self.assertRaises(ValueError): importance_summary([1,-1],.1)
        with self.assertRaises(ValueError): CoherentMPS([np.zeros((1,2,1))])

    def test_local_reductions_against_complete_dense_partial_trace(self):
        rng=np.random.default_rng(291005); dense=rng.normal(size=(3,3,3,3))+1j*rng.normal(size=(3,3,3,3)); dense/=np.linalg.norm(dense)
        tensors,_=dense_to_mps(dense,20); e=CoherentMPS(tensors); singles,pairs=local_reduced_states(e,[0,1,2])
        def reference(modes):
            rest=[i for i in range(4) if i not in modes]; matrix=dense.transpose(modes+rest).reshape(3**len(modes),-1); return matrix@matrix.conj().T
        for i in range(4): np.testing.assert_allclose(singles[i],reference([i]),atol=3e-15,rtol=0)
        for i in range(3): np.testing.assert_allclose(pairs[i],reference([i,i+1]),atol=3e-15,rtol=0)

    def test_integrated_effect_against_independent_dense_quadrature(self):
        rng=np.random.default_rng(291006)
        for shape,r in [((4,),2),((3,3),4)]:
            dense=rng.normal(size=shape)+1j*rng.normal(size=shape); dense/=np.linalg.norm(dense); L=rng.normal(size=(2*len(shape),r))*.5
            effect=displacement_vacuum_effect(L,shape); rho=np.outer(dense.ravel(),dense.ravel().conj())
            expected=quadrature_vacuum(dense,L)
            self.assertAlmostEqual(vacuum_expectation(rho,effect),expected,delta=4e-14)
            self.assertGreater(np.linalg.eigvalsh(effect)[0],-1e-14)

    def test_displaced_product_fock_two_mode_law(self):
        n=.3; L=np.sqrt(2*n)*np.eye(4); rho=np.zeros((9,9)); rho[1*3+2,1*3+2]=1
        q1=n/(1+n)**2; q2=n*n/(1+n)**3
        both=vacuum_expectation(rho,displacement_vacuum_effect(L,(3,3)))
        self.assertAlmostEqual(both,q1*q2,delta=2e-15)
        np.testing.assert_allclose(two_mode_threshold(q1,q2,both),[q1*q2,(1-q1)*q2,q1*(1-q2),(1-q1)*(1-q2)],atol=2e-15,rtol=0)


if __name__=='__main__': unittest.main()
