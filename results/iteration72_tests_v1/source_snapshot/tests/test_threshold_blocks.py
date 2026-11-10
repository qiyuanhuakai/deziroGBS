import functools,itertools,math,unittest
import numpy as np
from gbslab.threshold_blocks import contiguous_density,reduce_density,subset_effects,finite_block_law,gaussian_block_law,threshold_from_vacua,prefix_marginal,distance_diagnostics
from gbslab.displaced_vacuum import CoherentMPS
from gbslab.displaced_marginals import displacement_vacuum_effect
from gbslab.mps_probe import dense_to_mps
from test_displaced_sampling import dense_partial_trace,mixed_dense_law
from gbslab.exact import ThresholdModel


def bargmann_effect(noise,dimensions):
    """Independent complex-Q Gaussian Fock recurrence, normalized coefficients."""
    k=len(dimensions);C=np.asarray(noise)@np.asarray(noise).T;xx=C[:k,:k];pp=C[k:,k:];xp=C[:k,k:];px=C[k:,:k]
    normal=(xx+pp+1j*(px-xp))/4;anomalous=(xx-pp+1j*(xp+px))/4
    Q=np.eye(2*k)+np.block([[normal,anomalous],[anomalous.conj(),normal.conj()]]);swap=np.block([[np.zeros((k,k)),np.eye(k)],[np.eye(k),np.zeros((k,k))]])
    S=(np.eye(2*k)-np.linalg.inv(Q))@swap;det=np.linalg.det(Q)
    if abs(det.imag)>1e-10 or det.real<=0:raise ArithmeticError('Invalid complex Gaussian Q determinant')
    pref=1/math.sqrt(det.real)
    @functools.lru_cache(None)
    def coefficient(n):
        if not any(n):return 1.+0j
        if sum(n)%2:return 0j
        i=max(range(2*k),key=lambda j:n[j]);rest=list(n);rest[i]-=1;terms=[]
        for j,count in enumerate(rest):
            if count:
                child=rest.copy();child[j]-=1;terms.append(S[i,j]*math.sqrt(count/n[i])*coefficient(tuple(child)))
        return complex(math.fsum(x.real for x in terms),math.fsum(x.imag for x in terms))
    occupations=list(np.ndindex(dimensions));return pref*np.array([[coefficient(a+b) for b in occupations] for a in occupations])


class ThresholdBlockTests(unittest.TestCase):
    def test_contiguous_density_matches_dense_partial_trace(self):
        rng=np.random.default_rng(321001);dense=rng.normal(size=(3,)*6)+1j*rng.normal(size=(3,)*6);dense/=np.linalg.norm(dense);tensors,_=dense_to_mps(dense,100);evaluator=CoherentMPS(tensors)
        for start,length in ((0,4),(1,4),(2,4),(2,2)):
            rho,dims=contiguous_density(evaluator,start,length);np.testing.assert_allclose(rho,dense_partial_trace(dense,range(start,start+length)),atol=3e-15,rtol=0)

    def test_noncontiguous_reduction_and_trace(self):
        rng=np.random.default_rng(321002);dense=rng.normal(size=(2,3,4,2))+1j*rng.normal(size=(2,3,4,2));dense/=np.linalg.norm(dense);rho=np.outer(dense.ravel(),dense.ravel().conj())
        for sites in ((0,2),(1,3),(),(0,1,2,3)):np.testing.assert_allclose(reduce_density(rho,dense.shape,sites),dense_partial_trace(dense,sites),atol=3e-16,rtol=0)

    def test_bargmann_covariance_formula_against_real_wick(self):
        rng=np.random.default_rng(321003)
        for dimensions in ((3,),(3,3),(3,2,3),(2,2,2,2)):
            F=.4*rng.normal(size=(2*len(dimensions),5));np.testing.assert_allclose(bargmann_effect(F,dimensions),displacement_vacuum_effect(F,dimensions),atol=3e-15,rtol=0)

    def test_full_mixed_block_law_matches_independent_dense_law(self):
        rng=np.random.default_rng(321004);dense=rng.normal(size=(3,3,3))+1j*rng.normal(size=(3,3,3));dense/=np.linalg.norm(dense);F=.4*rng.normal(size=(6,3));rho=np.outer(dense.ravel(),dense.ravel().conj());p,_=finite_block_law(rho,dense.shape,subset_effects(F,dense.shape));expected=mixed_dense_law(dense,F)
        # The older dense helper is big-endian; this module indexes physical bit i.
        perm=[int(f'{i:03b}'[::-1],2) for i in range(8)];np.testing.assert_allclose(p,expected[perm],atol=2e-15,rtol=0)

    def test_gaussian_law_and_prefix_restriction(self):
        rng=np.random.default_rng(321005);F=.25*rng.normal(size=(10,4));V=np.eye(10)+F@F.T;p,_=gaussian_block_law(V);np.testing.assert_allclose(p,ThresholdModel(V).full_distribution(),atol=3e-15,rtol=0)
        q,_=gaussian_block_law(V[np.ix_([0,1,5,6],[0,1,5,6])]);np.testing.assert_allclose(prefix_marginal(p,2),q,atol=3e-15,rtol=0)

    def test_cauchy_block_upper_and_marginal_contraction(self):
        rng=np.random.default_rng(321006)
        for _ in range(20):
            p=rng.dirichlet(np.ones(16));q=rng.dirichlet(np.ones(16));d=distance_diagnostics(p,q);self.assertLessEqual(d['tv'],d['hellinger_cauchy_upper_for_this_block']+1e-15)
            self.assertLessEqual(distance_diagnostics(prefix_marginal(p,2),prefix_marginal(q,2))['tv'],d['tv']+1e-15)
        self.assertEqual(distance_diagnostics(np.array([1.,0]),np.array([0.,1.]))['hellinger_cauchy_upper_for_this_block'],1.)

    def test_invalid_laws_and_join_cap_reject(self):
        with self.assertRaises(ArithmeticError):threshold_from_vacua([1.,.1,.1,.9])
        with self.assertRaises(ValueError):threshold_from_vacua([1.,.2,.3])
        evaluator=CoherentMPS([np.ones((1,4,1))]*4)
        with self.assertRaises(MemoryError):contiguous_density(evaluator,0,4,maximum_elements=10)
        with self.assertRaises(ValueError):reduce_density(np.eye(4),(2,2),(1,0))


if __name__=='__main__':unittest.main()
