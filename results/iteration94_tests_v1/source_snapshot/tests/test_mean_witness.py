import itertools,math,unittest
from fractions import Fraction as F
from gbslab.mean_witness import dyadic_root_lower,product_mean_interval,integer_count_tail_upper


class MeanWitnessTests(unittest.TestCase):
    def test_root_adjacent_grid_certificate(self):
        for a,n in ((F(1,3200),96),(F(17,29),7),(F(1,1<<120),3),(F(81,256),4)):
            q=dyadic_root_lower(a,n);self.assertLessEqual(q**n,a);self.assertGreater((q+F(1,1<<40))**n,a)
        self.assertEqual(dyadic_root_lower(0,96),0);self.assertEqual(dyadic_root_lower(1,96),1)

    def test_constant_zero_one_and_failure_mapping(self):
        for x in (F(0),F(1,3),F(1)):
            r=product_mean_interval([x]*8,F(1,100));self.assertLessEqual(r['lower'],x);self.assertGreaterEqual(r['upper'],x)
        r=product_mean_interval([F(1,2),F(0),F(3,4)],F(1,100));self.assertEqual(r['lower'],0);self.assertEqual(r['attempts'],3)

    def test_exact_small_law_coverage(self):
        n=4;alpha=F(1,10);xs=(F(0),F(1,3),F(1));compositions=[(i,j,n-i-j) for i in range(n+1) for j in range(n-i+1)]
        limits={c:product_mean_interval([x for x,k in zip(xs,c) for _ in range(k)],alpha,bits=12) for c in compositions}
        for a in range(5):
            for b in range(5-a):
                ps=(F(a,4),F(b,4),F(4-a-b,4));mu=sum(p*x for p,x in zip(ps,xs));low=high=joint=F(0);mass=F(0)
                for c in compositions:
                    probability=F(math.factorial(n),math.prod(math.factorial(k) for k in c))*math.prod(p**k for p,k in zip(ps,c));mass+=probability;r=limits[c];badlo=r['lower']>mu;badhi=r['upper']<mu;low+=probability*badlo;high+=probability*badhi;joint+=probability*(badlo or badhi)
                self.assertEqual(mass,1);self.assertLessEqual(low,alpha);self.assertLessEqual(high,alpha);self.assertLessEqual(joint,2*alpha)

    def test_bernoulli_near_markov_boundary(self):
        alpha=F(1,10);n=3;q=dyadic_root_lower(alpha,n,bits=32);mu=q-F(1,1<<33);r=product_mean_interval([1]*n,alpha,bits=32);self.assertGreater(r['lower'],mu);self.assertLessEqual(mu**n,alpha);self.assertGreater(mu**n,F(99,100)*alpha)

    def test_all_integer_polynomial_majorants(self):
        for m in range(1,9):
            for k in range(1,m+1):
                for a in range(-1,k-1):
                    for x in range(m+1):self.assertGreaterEqual(F((x-a)*(x-a-1),(k-a)*(k-a-1)),int(x>=k))

    def test_discrete_mixture_tail_bounds(self):
        p=(F(1,5),F(1,10),F(1,10),F(1,5),F(2,5));mu=sum(j*x for j,x in enumerate(p));s=sum(j*j*x for j,x in enumerate(p))
        for k in range(-1,7):
            r=integer_count_tail_upper(k,4,(mu,mu),(s,s));true=sum(x for j,x in enumerate(p) if j>=k);self.assertGreaterEqual(r['upper'],true);self.assertLessEqual(r['upper'],1)
        self.assertEqual(integer_count_tail_upper(1,4,(0,0),(0,0))['upper'],0)

    def test_uncertain_moment_direction(self):
        r=integer_count_tail_upper(3,5,(F(1),F(2)),(F(4),F(6)));self.assertEqual(r['candidates'][0]['upper'],F(8,12));self.assertEqual(r['candidates'][2]['upper'],F(5,2))

    def test_invalid_inputs(self):
        for args in ((-1,2),(F(1,2),0),(F(1,2),True)):
            with self.assertRaises(ValueError):dyadic_root_lower(*args)
        for values,alpha in (([],F(1,10)),([F(3,2)],F(1,10)),([F(1,2)],0)):
            with self.assertRaises(ValueError):product_mean_interval(values,alpha)
        with self.assertRaises(ValueError):integer_count_tail_upper(2,4,(3,2),(1,2))
        with self.assertRaises(ValueError):integer_count_tail_upper(3,4,(3,3),(0,0))

if __name__=='__main__':unittest.main()
