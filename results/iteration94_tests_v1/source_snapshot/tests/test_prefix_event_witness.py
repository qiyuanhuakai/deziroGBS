import itertools
import unittest
from fractions import Fraction as F
from gbslab.prefix_event_witness import sparse_events, split_gain_interval, prefix_split_witness


class PrefixEventWitnessTests(unittest.TestCase):
    def test_gain_rectangle_extrema_exhaustive(self):
        intervals = [(F(a), F(b)) for a in range(-3, 4) for b in range(a, 4)]
        for x, y in itertools.product(intervals, repeat=2):
            samples = [(abs(a)+abs(b)-abs(a+b))/2
                       for a in (*x, sum(x)/2) for b in (*y, sum(y)/2)]
            self.assertEqual(split_gain_interval(x, y), (min(samples), max(samples)))

    def test_hidden_difference_revealed(self):
        q = [F(3, 8), F(1, 8), F(1, 8), F(3, 8)]
        r = prefix_split_witness(q, [(F(1, 4), F(1, 4))]*4)
        self.assertEqual(r['gain_interval'], (F(1, 4), F(1, 4)))
        self.assertEqual(r['coarse_midpoint_half_l1'], 0)
        self.assertEqual(r['certified_opposite_parents'], 2)

    def test_same_sign_no_gain(self):
        q = [F(3, 8), F(1, 8)]*2
        self.assertEqual(prefix_split_witness(q, [(F(1, 4), F(1, 4))]*4)['gain_interval'], (0, 0))

    def test_sparse_budgets_match_exhaustive_subsets(self):
        q = [F(1, 2), F(1, 3), F(1, 6), F(0)]
        p = [(F(1, 5), F(3, 10))]*4
        for row in sparse_events(q, p, (1, 2, 3, 4)):
            if row['budget'] == 'all': continue
            values = []
            for k in range(row['budget']+1):
                for subset in itertools.combinations(range(4), k):
                    values.append(sum(q[i]-p[i][1] if row['side']=='program_excess' else p[i][0]-q[i] for i in subset))
            self.assertEqual(row['TV_lower'], max(values))

    def test_equal_laws_empty_events(self):
        for row in sparse_events([F(1, 2)]*2, [(F(1, 2), F(1, 2))]*2):
            self.assertEqual(row['masks'], [])
            self.assertEqual(row['TV_lower'], 0)

    def test_ties_use_atom_index(self):
        rows = sparse_events([F(1, 2), F(1, 2), 0, 0], [(F(1, 4), F(1, 4))]*4, (1,))
        self.assertEqual(rows[0]['masks'], [0])
        self.assertEqual(rows[2]['masks'], [2])

    def test_uncertain_sign_contains_exact_gain(self):
        q = [F(3, 10), F(1, 5), F(1, 5), F(3, 10)]
        r = prefix_split_witness(q, [(F(1, 5), F(3, 10))]*4)
        self.assertEqual(r['gain_interval'], (0, F(1, 5)))

    def test_invalid_laws_and_budget(self):
        for q, p in (([1, 1], [(0, 1)]*2), ([1, 0], [(0, F(1, 4))]*2),
                     ([F(1, 3)]*3, [(0, 1)]*3), ([1, 0], [(1, 0)]*2)):
            with self.assertRaises(ValueError): sparse_events(q, p)
        with self.assertRaises(ValueError): sparse_events([1, 0], [(0, 1)]*2, (0,))
        with self.assertRaises(ValueError): split_gain_interval((1, 0), (0, 1))


if __name__ == '__main__': unittest.main()
