from fractions import Fraction as F
from itertools import combinations
import unittest
import mpmath as mp
import numpy as np
from gbslab.atom_certificate import selected_atoms
from gbslab.coupling_audit import audit,normalized_rationals
from gbslab.rational_gaussian import VacuumEnclosure,reciprocal_sqrt,interval_determinant,outward


def determinant(matrix):
    a=[[F(x) for x in row] for row in matrix];n=len(a);sign=1
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:return F(0)
        if pivot!=j:a[j],a[pivot]=a[pivot],a[j];sign=-sign
        for i in range(j+1,n):
            ratio=a[i][j]/a[j][j]
            for k in range(j,n):a[i][k]-=ratio*a[j][k]
    result=F(sign)
    for i in range(n):result*=a[i][i]
    return result


class AtomCertificateTests(unittest.TestCase):
    def test_selected_bounds_converge_to_full_failure_aware_tv(self):
        p,_=normalized_rationals(list(range(1,17)));oracle=lambda h:None if h==(1,0,1) else F(2+sum(h),9);full=audit(p,oracle);previous=F(1)
        for k in range(5):
            patterns=[c for size in range(k+1) for c in combinations(range(4),size)];target=lambda c:(p[sum(1<<j for j in c)],)*2;r=selected_atoms(4,patterns,oracle,target)
            self.assertLessEqual(r['tv_upper'],previous);previous=r['tv_upper'];self.assertLessEqual(r['tv_lower'],full['extended_tv']);self.assertGreaterEqual(r['tv_upper'],full['extended_tv'])
        self.assertEqual(r['tv_lower'],full['extended_tv']);self.assertEqual(r['tv_upper'],full['extended_tv']);self.assertEqual(r['seen_failure_mass'],full['failure_mass_seen'])

    def test_uncertain_target_intervals_and_unvisited_failure(self):
        p=[F(1,4)]*4;oracle=lambda h:None if h==(1,) else F(1,2);full=audit(p,oracle)
        r=selected_atoms(2,[()],oracle,lambda c:(F(1,5),F(3,10)));self.assertEqual(r['seen_failure_mass'],0);self.assertLessEqual(r['tv_lower'],full['extended_tv']);self.assertGreaterEqual(r['tv_upper'],full['extended_tv'])
        r=selected_atoms(2,[(),(0,)],oracle,lambda c:(F(1,5),F(3,10)));self.assertEqual(r['seen_failure_mass'],F(1,2));self.assertLessEqual(r['tv_lower'],full['extended_tv'])

    def test_large_product_without_target_table(self):
        m=144;q=F(1,1000);target=lambda c:(q**len(c)*(1-q)**(m-len(c)),)*2;r=selected_atoms(m,[()]+[(j,) for j in range(m)],lambda h:q,target)
        self.assertEqual(r['queries'],m*(m+1)//2);self.assertEqual(r['tv_lower'],0);self.assertEqual(r['tv_upper'],1-(1-q)**m-m*q*(1-q)**(m-1))

    def test_validation_program_failure_and_grid(self):
        for patterns in ([(),()],[(1,0)],[(2,)]):
            with self.assertRaises(ValueError):selected_atoms(2,patterns,lambda h:F(1,2),lambda c:(0,1))
        with self.assertRaises(ValueError):selected_atoms(1,[()],lambda h:2,lambda c:(0,1))
        def broken(h):raise RuntimeError('not numeric failure')
        with self.assertRaises(RuntimeError):selected_atoms(2,[()],broken,lambda c:(0,1))
        r=selected_atoms(1,[(),(0,)],lambda h:F(1,3),lambda c:(F(1,2),F(1,2)),uniform_bits=3);self.assertEqual(r['proxy_masses'],(F(5,8),F(3,8)))

    def test_exact_determinants_and_high_precision_atoms(self):
        A=np.array([[1,0,2,1],[0,2,1,0],[1,1,0,1],[0,1,1,2]],float)/8;V=np.eye(4)+A@A.T;certificate=VacuumEnclosure(V);Q=[[(F(float(V[i,j]))+int(i==j))/2 for j in range(4)] for i in range(4)]
        complement={}
        for removed in ((),(0,),(1,),(0,1)):
            indices=[j for j in range(4) if j%2 not in removed];D=determinant([[Q[i][j] for j in indices] for i in indices]);lo,hi=certificate.complement(removed);self.assertLessEqual(lo*lo*D,1);self.assertGreaterEqual(hi*hi*D,1);complement[removed]=D
        with mp.workdps(100):
            convert=lambda x:mp.mpf(x.numerator)/x.denominator
            for clicks in ((),(0,),(1,),(0,1)):
                exact=mp.fsum((-1)**(len(clicks)-len(c))/mp.sqrt(convert(complement[c])) for k in range(len(clicks)+1) for c in combinations(clicks,k));lo,hi=certificate.atom(clicks);self.assertLessEqual(convert(lo),exact);self.assertGreaterEqual(convert(hi),exact)

    def test_exact_input_addition_preserved_near_vacuum(self):
        v=np.nextafter(1.,2.);r=VacuumEnclosure(v*np.eye(4));expected=(2/(F(float(v))+1))**2;lo,hi=r.vacuum;self.assertLessEqual(lo,expected);self.assertGreaterEqual(hi,expected);self.assertGreater(r.delta,0)
        r=VacuumEnclosure(np.eye(4));self.assertEqual(r.vacuum,(1,1));self.assertEqual(r.atom((0,)),(0,0))

    def test_interval_determinant_and_square_root(self):
        rng=np.random.default_rng(341001)
        for n in (2,4):
            center=[[F(int(rng.integers(-8,9)),8) for j in range(n)] for i in range(n)];box=[[(v-F(1,16),v+F(1,16)) for v in row] for row in center];lo,hi=interval_determinant(box)
            for _ in range(12):
                sample=[[v+F(int(rng.integers(-1,2)),16) for v in row] for row in center];d=determinant(sample);self.assertLessEqual(lo,d);self.assertGreaterEqual(hi,d)
        for x in (F(1),F(2),F(1,10**40),F(17,29)):
            lo,hi=reciprocal_sqrt((x,x));self.assertLessEqual(lo*lo*x,1);self.assertGreaterEqual(hi*hi*x,1)
        lo,hi=outward(F(-1,3),F(1,7));self.assertLessEqual(lo,F(-1,3));self.assertGreaterEqual(hi,F(1,7))

    def test_invalid_covariance_not_repaired(self):
        with self.assertRaises(ValueError):VacuumEnclosure(np.array([[1.,.1],[0.,1.]]))
        with self.assertRaises(np.linalg.LinAlgError):VacuumEnclosure(-2*np.eye(2))
        with self.assertRaises(ValueError):VacuumEnclosure(np.eye(3))

    def test_positive_Q_does_not_certify_quantum_physicality(self):
        r=VacuumEnclosure(.5*np.eye(2))
        self.assertTrue(r.diagnostics['Q_positive_definite_proved'])
        self.assertTrue(r.diagnostics['quantum_uncertainty_not_certified'])
        self.assertGreater(r.vacuum[0],1)
        self.assertLess(r.atom((0,))[1],0)
        with self.assertRaises(ValueError):selected_atoms(1,[()],lambda h:F(1,2),r.atom)

if __name__=='__main__':unittest.main()
