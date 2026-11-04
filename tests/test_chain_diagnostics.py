import unittest
import numpy as np
from gbslab.phase2 import chain_diagnostics
from gbslab.surrogates import distribution_metrics


class ChainDiagnosticTests(unittest.TestCase):
    def test_correlated_mixture_and_reversed_asymmetry(self):
        p=np.array([.25,.05,.05,.15,.05,.15,.25,.05])
        q=np.array([.05,.15,.2,.1,.1,.1,.2,.1])
        for target,model in [(p,q),(q,p)]:
            d=chain_diagnostics(target,model);m=distribution_metrics(target,model)
            self.assertAlmostEqual(d['sum_expected_conditional_kl'],m['kl'],places=12)
            self.assertGreaterEqual(d['tv_hybrid_bound']+1e-14,m['tv'])

    def test_parity_identity(self):
        p=np.array([.25,0,0,.25,0,.25,.25,0]);q=np.full(8,.125)
        d=chain_diagnostics(p,q)
        self.assertAlmostEqual(d['tv_hybrid_bound'],.5)
        self.assertAlmostEqual(d['sum_expected_conditional_kl'],np.log(2))

    def test_zero_prefix_extension_is_explicit(self):
        p=np.full(4,.25);q=np.array([1.,0,0,0])
        d=chain_diagnostics(p,q)
        self.assertTrue(np.isinf(d['sum_expected_conditional_kl']))
        self.assertGreater(d['stages'][1]['p_mass_on_q_zero_prefixes'],0)
        self.assertGreaterEqual(d['tv_hybrid_bound'],.75)

    def test_tiny_positive_outcome_is_not_rounded_to_zero(self):
        p=np.array([.5,.5]);q=np.array([1e-20,1.0])
        d=chain_diagnostics(p,q)
        self.assertTrue(np.isfinite(d['sum_expected_conditional_kl']))
        self.assertAlmostEqual(d['sum_expected_conditional_kl'],
                               distribution_metrics(p,q)['kl'],places=12)

    def test_stage_zero_is_least_significant_mode(self):
        # P has bit0 always zero and bit1 fair; Q is uniform.
        p=np.array([.5,0,.5,0]);q=np.full(4,.25)
        d=chain_diagnostics(p,q)
        self.assertAlmostEqual(d['stages'][0]['weighted_conditional_tv'],.5)
        self.assertAlmostEqual(d['stages'][1]['weighted_conditional_tv'],0)

    def test_accepted_normalization_residue_matches_metric_convention(self):
        p=np.array([.5,.5])*(1+5e-10);q=np.array([1e-20,1.0])
        d=chain_diagnostics(p,q)
        self.assertAlmostEqual(d['sum_expected_conditional_kl'],
                               distribution_metrics(p,q)['kl'],places=12)


if __name__=='__main__':unittest.main()
