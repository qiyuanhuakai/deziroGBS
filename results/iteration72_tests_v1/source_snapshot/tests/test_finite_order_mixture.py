import unittest
from fractions import Fraction as F
from itertools import product
from gbslab.finite_order_mixture import select_order,selector_law,mixture_law,sample_order_mixture


class FiniteOrderMixtureTests(unittest.TestCase):
    def test_exhaustive_small_word_partition(self):
        for bits in range(1,9):
            for n in range(1,min(9,1<<bits)+1):
                counts=[0]*n
                for w in range(1<<bits):counts[select_order(w,n,bits=bits)]+=1
                r=selector_law(n,bits=bits);self.assertEqual(tuple(counts),r['counts']);self.assertEqual(sum(r['weights']),1)

    def test_all_native_partition_boundaries(self):
        for n in (1,2,3,5,17,64):
            r=selector_law(n);b=r['boundaries']
            for j in range(n):
                self.assertEqual(select_order(b[j],n),j);self.assertEqual(select_order(b[j+1]-1,n),j)
                self.assertLessEqual(j*(1<<64),n*b[j]);self.assertLess(n*(b[j+1]-1),(j+1)*(1<<64))

    def test_three_way_bias_is_exact(self):
        q,r=divmod(1<<64,3);self.assertEqual(r,1);law=selector_law();self.assertEqual(law['counts'],(q+1,q,q));self.assertEqual(law['uniform_TV'],F(2,3*(1<<64)))

    def test_mixture_matches_full_random_experiment(self):
        components=((F(1,4),F(3,4)),(F(1,2),F(1,2)),(F(1),F(0)));hist=[0,0]
        for word,coin in product(range(8),range(4)):
            c=select_order(word,3,bits=3);hist[0 if coin<4*components[c][0] else 1]+=1
        r=mixture_law(components,bits=3);self.assertEqual(r['distribution'],tuple(F(h,32) for h in hist));self.assertEqual(r['ideal_distribution'],(F(7,12),F(5,12)))

    def test_sharp_bias_bound_for_disjoint_components(self):
        laws=((1,0,0),(0,1,0),(0,0,1));r=mixture_law(laws);self.assertEqual(r['ideal_distance'],selector_law()['uniform_TV'])

    def test_identical_components_remove_selector_bias(self):
        p=(F(1,7),F(6,7));r=mixture_law((p,p,p));self.assertEqual(r['distribution'],p);self.assertEqual(r['ideal_distance'],0)

    def test_dispatch_restores_both_orders_and_skips_empty(self):
        calls=[]
        def a(n):calls.append((0,n));return ['10','01'][:n]
        def b(n):calls.append((1,n));return ['11']*n
        def c(n):calls.append((2,n));return ['10']*n
        r=sample_order_mixture([7,0,7,1],[a,b,c],[(0,1),(0,1),(1,0)],bits=3)
        self.assertEqual(r['choices'],(2,0,2,0));self.assertEqual(calls,[(0,2),(2,2)]);self.assertEqual(r['outputs'],('01','10','01','01'));self.assertEqual(r['component_indices'],(0,0,1,1))

    def test_empty_request_never_calls_components(self):
        def fail(n):raise AssertionError('unused callback')
        r=sample_order_mixture([], [fail]*3,[(0,1)]*3);self.assertEqual(r['outputs'],());self.assertEqual(r['component_counts'],(0,0,0))

    def test_bad_words_are_rejected_before_callbacks(self):
        calls=[]
        for word in (-1,1<<64,True,1.5):
            with self.assertRaises(ValueError):sample_order_mixture([0,word],[lambda n:calls.append(n)]*3,[(0,1)]*3)
        self.assertFalse(calls)

    def test_rejects_malformed_components_and_permutations(self):
        for p in (((0,0),)*3,((True,1),)*3,((0,1),(0,1)),((),)*3):
            with self.assertRaises(ValueError):sample_order_mixture([0],[lambda n:['00']*n]*3,p)
        for callback in (lambda n:[],lambda n:['012'],lambda n:[12],lambda n:['0']):
            with self.assertRaises(ValueError):sample_order_mixture([0],[callback],[(0,1)])

    def test_invalid_laws_and_sizes(self):
        for p in ([],[(F(1,2),)],[(1,0),(1,)],[(2,-1)], [()]):
            with self.assertRaises(ValueError):mixture_law(p)
        for n,b in ((0,64),(65,64),(3,1),(True,64),(3,0),(3,65),(3,True)):
            with self.assertRaises(ValueError):selector_law(n,bits=b)

    def test_permutation_sensitive_observable(self):
        r=sample_order_mixture([0,3,7],[lambda n:['100']*n]*3,[(0,1,2),(1,2,0),(2,0,1)],bits=3)
        self.assertEqual(r['outputs'],('100','010','001'));self.assertEqual([s.count('1') for s in r['outputs']],[1,1,1])
