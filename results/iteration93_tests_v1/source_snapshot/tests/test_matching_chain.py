import unittest
from collections import Counter
from fractions import Fraction as F
from itertools import product
from gbslab.matching_chain import MatchingChain,uniform_below,rational_bernoulli

class MatchingChainTests(unittest.TestCase):
    def test_single_edge_kernel(self):
        c=MatchingChain(2,[(0,1)],[2]);s=c.states()
        self.assertEqual(s,(0,1));self.assertEqual(c.transition(s),[[F(1,2),F(1,2)],[F(1,4),F(3,4)]])
    def test_empty_matching_weight(self):
        self.assertEqual(MatchingChain(2,[(0,1)],[F(2,3)]).weight(0),1)
    def test_deletion(self):
        self.assertEqual(MatchingChain(2,[(0,1)],[3]).proposal(1,0),(0,F(1,3),'delete'))
    def test_addition(self):
        self.assertEqual(MatchingChain(2,[(0,1)],[F(1,3)]).proposal(0,0),(1,F(1,3),'add'))
    def test_swap_weight_ratio(self):
        c=MatchingChain(3,[(0,1),(1,2)],[6,2]);self.assertEqual(c.proposal(1,1),(2,F(1,3),'swap'))
    def test_double_conflict_stays(self):
        c=MatchingChain(4,[(0,1),(1,2),(2,3)],[1,1,1]);self.assertEqual(c.proposal(5,1),(5,F(1),'blocked'))
    def test_triangle_state_set(self):
        self.assertEqual(MatchingChain(3,[(0,1),(0,2),(1,2)],[1,1,1]).states(),(0,1,2,4))
    def test_weighted_detailed_balance(self):
        c=MatchingChain(4,[(0,1),(0,3),(1,2),(2,3)],[F(1,3),2,4,F(2,5)]);s=c.states();p=c.transition(s)
        for i,a in enumerate(s):
            for j,b in enumerate(s):self.assertEqual(c.weight(a)*p[i][j],c.weight(b)*p[j][i])
    def test_quantized_acceptance_bias(self):
        c=MatchingChain(2,[(0,1)],[3]);s=c.states();p=c.transition(s,bits=8)
        self.assertNotEqual(F(1,4)*p[0][1],F(3,4)*p[1][0])
    def test_edge_selector_bias(self):
        c=MatchingChain(3,[(0,1),(0,2),(1,2)],[1,1,1]);p=c.transition(c.states(),bits=8)
        self.assertNotEqual(p[1][2],p[2][1])
    def test_small_word_full_transition(self):
        c=MatchingChain(3,[(0,1),(0,2),(1,2)],[1,3,F(1,3)]);states=c.states();p=c.transition(states,bits=2)
        for i,s in enumerate(states):
            counts=Counter(c.step(s,iter(words).__next__,mode='naive',bits=2) for words in product(range(4),repeat=3))
            self.assertEqual([F(counts[x],64) for x in states],p[i])
    def test_forced_rejection_and_residues(self):
        words=iter([15,14]);self.assertEqual(uniform_below(3,words.__next__,bits=4),2)
        counts=Counter(uniform_below(3,iter([w]).__next__,bits=4) for w in range(15));self.assertEqual(set(counts.values()),{5})
    def test_rational_bernoulli_rejection(self):
        self.assertTrue(rational_bernoulli(F(2,3),iter([15,13]).__next__,bits=4))
        self.assertFalse(rational_bernoulli(F(2,3),iter([14]).__next__,bits=4))
    def test_exact_step_variable_consumption(self):
        c=MatchingChain(3,[(0,1),(0,2),(1,2)],[1,1,1]);words=iter([0,15,14]);self.assertEqual(c.step(0,words.__next__,bits=4),4)
        with self.assertRaises(StopIteration):next(words)
        self.assertEqual(c.step(0,iter([1]).__next__,bits=4),0)
    def test_invalid_input(self):
        for edges,weights in [([],[]),([(1,0)],[1]),([(0,1),(0,1)],[1,1]),([(0,1)],[0])]:
            with self.assertRaises(ValueError):MatchingChain(2,edges,weights)
        c=MatchingChain(3,[(0,1),(1,2)],[1,1])
        with self.assertRaises(ValueError):c.step(3,lambda:0)
        with self.assertRaises(ValueError):c.step(0,lambda:-1)
        with self.assertRaises(ValueError):uniform_below(17,lambda:0,bits=4)
        with self.assertRaises(ValueError):rational_bernoulli(2,lambda:0)
    def test_large_chain_local_without_enumeration(self):
        c=MatchingChain(20,[(i,i+1) for i in range(19)],[1]*19)
        self.assertEqual(c.step(0,iter([0,0]).__next__),1)
        with self.assertRaises(ValueError):c.states()

if __name__=='__main__':unittest.main()
