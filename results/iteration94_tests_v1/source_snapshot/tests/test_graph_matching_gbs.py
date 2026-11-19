import unittest
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from gbslab.graph_matching_gbs import graph,subset_counts,lifted_structure,partition_by_size,perfect_mass,least_half_multiplier,categorical,draw_word,determinant,gaussian_bridge


class GraphMatchingTests(unittest.TestCase):
    def test_empty_graph(self):
        self.assertEqual(subset_counts(4,[]),(1,)+(0,)*15)
        st=lifted_structure(4,[])
        self.assertEqual(st['coefficients'],((0,0,1),(1,0,4),(2,0,6),(3,0,4),(4,0,1)))
        self.assertEqual(len(st['perfect_matchings']),1)

    def test_single_edge_polynomial(self):
        st=lifted_structure(2,[(0,1)])
        self.assertEqual(st['coefficients'],((0,0,1),(1,0,2),(1,1,2),(2,0,1),(2,2,1)))
        self.assertEqual(partition_by_size(2,st['coefficients'],F(1,3)),(1,F(20,9),F(82,81)))

    def test_complete_six_counts(self):
        pm=subset_counts(6,combinations(range(6),2))
        for s,p in enumerate(pm):self.assertEqual(p,{0:1,2:1,4:3,6:15}.get(s.bit_count(),0))

    def test_disjoint_factorization(self):
        pm=subset_counts(4,[(0,1),(2,3)])
        self.assertEqual({s:p for s,p in enumerate(pm) if p},{0:1,3:1,12:1,15:1})
        c=F(2,3);self.assertEqual(sum(p*p*c**(2*s.bit_count()) for s,p in enumerate(pm)),(1+c**4)**2)

    def test_relabelling(self):
        edges=[(0,1),(1,2),(2,3),(1,4)];perm=[3,0,4,2,1]
        a=subset_counts(5,edges);b=subset_counts(5,[(perm[i],perm[j]) for i,j in edges])
        for s,x in enumerate(a):self.assertEqual(x,b[sum(1<<perm[j] for j in range(5) if s&(1<<j))])

    def test_lift_witness_integrity(self):
        st=lifted_structure(4,combinations(range(4),2));counts=Counter()
        for w in st['perfect_matchings']:
            self.assertEqual(sorted(v for e in w['edges'] for v in e),list(range(8)))
            self.assertTrue(all(e in st['lifted_edges'] for e in w['edges']));counts[w['subset']]+=1
        self.assertEqual(counts[15],9);self.assertEqual(sum(counts.values()),16)

    def test_all_matchings_edge_enumeration(self):
        st=lifted_structure(3,[(0,1),(1,2)]);counts=Counter()
        for k in range(4):
            for selected in combinations(st['lifted_edges'],k):
                if len(set(v for e in selected for v in e))==2*k:
                    counts[k,sum((a<3)==(b<3) for a,b in selected)]+=1
        self.assertEqual(counts,{(k,h):v for k,h,v in st['coefficients']})

    def test_minimal_reweighting(self):
        z=(F(1),F(4),F(2));alpha=least_half_multiplier(z)
        self.assertEqual(alpha,3);self.assertLess(perfect_mass(z,alpha-1),F(1,2));self.assertGreaterEqual(perfect_mass(z,alpha),F(1,2))

    def test_finite_projection_counterexample(self):
        p=[F(1,5),F(1,5),F(3,5)];labels=[0,1,0]
        natural=categorical(p,bits=2);self.assertEqual(natural['counts'],(1,1,2))
        projected=[sum(k for s,k in zip(labels,natural['counts']) if s==i) for i in range(2)]
        direct=categorical([F(4,5),F(1,5)],bits=2)
        self.assertEqual(projected,[3,1]);self.assertEqual(direct['counts'],(4,0))
        self.assertLess(abs(F(projected[0],4)-F(4,5)),abs(F(direct['counts'][0],4)-F(4,5)))

    def test_grouping_telescopes(self):
        p=[F(1,5),F(3,5),F(1,5)];labels=[0,0,1]
        for bits in range(1,9):
            grouped=categorical(p,bits=bits);direct=categorical([F(4,5),F(1,5)],bits=bits)
            for word in range(1<<bits):self.assertEqual(labels[draw_word(grouped,word)],draw_word(direct,word))

    def test_exhaustive_small_words(self):
        p=[F(0),F(1,7),F(0),F(2,7),F(4,7),F(0)]
        for bits in range(1,9):
            table=categorical(p,bits=bits);counts=Counter(draw_word(table,w) for w in range(1<<bits))
            self.assertEqual(tuple(counts[i] for i in range(len(p))),table['counts'])

    def test_exact_boundary(self):
        table=categorical([F(0),F(1,2),F(0),F(1,2)],bits=3)
        self.assertEqual([draw_word(table,w) for w in (0,3,4,7)],[1,1,3,3])

    def test_gaussian_single_edge(self):
        c=F(1,4);b=gaussian_bridge(2,[(0,1)],c,1+c**4)
        self.assertEqual(b['collision_free_probability'],(1-c**8,1-c**8))

    def test_gaussian_scope(self):
        self.assertEqual(gaussian_bridge(1,[],F(2),F(1))['collision_free_probability'],(1,1))
        self.assertEqual(gaussian_bridge(2,[(0,1)],F(1),F(2))['status'],'invalid')
        self.assertEqual(gaussian_bridge(3,[(0,1),(1,2)],F(3,4),F(1))['status'],'uncertified')

    def test_determinant_pivot_and_singular(self):
        self.assertEqual(determinant([[0,2],[3,4]]),-6)
        self.assertEqual(determinant([[1,2],[2,4]]),0)

    def test_invalid_inputs(self):
        for n,edges in [(0,[]),(9,[]),(2,[(0,0)]),(2,[(0,2)]),(2,[(0,1),(1,0)])]:
            with self.assertRaises(ValueError):graph(n,edges)
        for p in [[],[-1,2],[F(1,2)]]:
            with self.assertRaises(ValueError):categorical(p)
        for word in (-1,256,True):
            with self.assertRaises(ValueError):draw_word(categorical([1],bits=8),word)
        with self.assertRaises(ValueError):categorical([1],bits=65)
        with self.assertRaises(ValueError):partition_by_size(1,[(0,0,1)],0)

if __name__=='__main__':unittest.main()
