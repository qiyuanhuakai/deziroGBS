from fractions import Fraction as F
import itertools,unittest
import numpy as np
from gbslab.coupling_audit import audit,normalized_rationals,uniform_grid_probability

def independent_law(m,oracle,bits=None):
    law=[F(0)]*(1<<m);failure=F(0)
    def visit(prefix,weight):
        nonlocal failure
        if len(prefix)==m:law[sum(b<<i for i,b in enumerate(prefix))]+=weight;return
        q=oracle(prefix)
        if q is None:failure+=weight;return
        if bits is not None:q=F(sum(F(i,1<<bits)<q for i in range(1<<bits)),1<<bits)
        visit(prefix+(0,),weight*(1-q));visit(prefix+(1,),weight*q)
    visit((),F(1));return law,failure

class CouplingAuditTests(unittest.TestCase):
    def test_equal_laws_zero_bound_and_late_failure_exact_bound(self):
        p=[F(1,16)]*16
        same=audit(p,lambda prefix:F(1,2));self.assertEqual(same['coupling_upper'],0)
        failed=audit(p,lambda prefix:None if prefix==(0,0,0) else F(1,2));self.assertEqual(failed['coupling_upper'],F(1,8));self.assertEqual(failed['extended_tv'],F(1,8));self.assertEqual(failed['failure_mass_seen'],F(1,8))

    def test_root_failure_and_support_mismatch(self):
        p=[F(1,4)]*4;failed=audit(p,lambda prefix:None);self.assertEqual(failed['extended_tv'],1)
        zeros=audit(p,lambda prefix:F(0));self.assertEqual(zeros['extended_tv'],F(3,4));self.assertEqual(zeros['coupling_upper'],F(3,4))
        outside=audit([F(1),F(0),F(0),F(0)],lambda prefix:F(1,2));self.assertEqual(outside['extended_tv'],F(3,4))

    def test_random_rational_trees_against_independent_enumeration(self):
        rng=np.random.default_rng(331001)
        for _ in range(24):
            p,_=normalized_rationals(rng.integers(1,20,size=16).tolist());table={prefix:(None if len(prefix)>1 and rng.random()<.15 else F(int(rng.integers(0,9)),8)) for length in range(4) for prefix in itertools.product((0,1),repeat=length)};oracle=lambda prefix:table[prefix]
            expected,failure=independent_law(4,oracle);result=audit(p,oracle);self.assertEqual(list(result['distribution']),expected);self.assertEqual(result['failure_mass_seen'],failure)

    def test_partial_work_retains_unresolved_mass_and_monotone_bound(self):
        p=[F(1,16)]*16;oracle=lambda prefix:F(2,5);whole=audit(p,oracle);previous=F(1)
        for budget in range(16):
            partial=audit(p,oracle,maximum_queries=budget);self.assertLessEqual(partial['coupling_upper'],previous);self.assertGreaterEqual(partial['coupling_upper'],whole['extended_tv']);previous=partial['coupling_upper']
            if not partial['complete']:self.assertIsNone(partial['extended_tv'])
        self.assertEqual(audit(p,oracle,maximum_queries=3)['coupling_upper'],1)

    def test_uniform_grid_exhaustive_and_subgrid_positive_branch(self):
        for q in (F(0),F(1,1000),F(1,8),F(1,3),F(7,8),F(1)):
            self.assertEqual(uniform_grid_probability(q,3),F(sum(F(i,8)<q for i in range(8)),8))
        self.assertEqual(uniform_grid_probability(F(1,10**20),53),F(1,1<<53))
        p=[F(1,8)]*8;oracle=lambda prefix:F(1,3);expected,failure=independent_law(3,oracle,bits=3);result=audit(p,oracle,uniform_bits=3);self.assertEqual(list(result['distribution']),expected)

    def test_order_mapping_and_explicit_normalization(self):
        p,mass=normalized_rationals([.1,.2,.3,.4]);self.assertEqual(sum(p),1);self.assertNotEqual(mass,0)
        def oracle(prefix):return F(1,3) if not prefix else F(3,4)
        result=audit(p,oracle,order=(1,0));self.assertEqual(result['distribution'],(F(1,6),F(1,2),F(1,12),F(1,4)))

    def test_program_errors_propagate_and_invalid_laws_reject(self):
        with self.assertRaises(ValueError):audit([.2,.2],lambda p:F(1,2))
        with self.assertRaises(ValueError):audit([F(1,2)]*2,lambda p:F(3,2))
        def broken(prefix):raise RuntimeError('program failure')
        with self.assertRaisesRegex(RuntimeError,'program failure'):audit([F(1,2)]*2,broken)

if __name__=='__main__':unittest.main()
