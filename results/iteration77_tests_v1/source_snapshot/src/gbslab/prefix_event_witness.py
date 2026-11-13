"""Exact cylinder-event witnesses and information gained by splitting a bit."""
from fractions import Fraction as F


def checked_laws(program, target):
    q = [F(x) for x in program]
    p = [(F(a), F(b)) for a, b in target]
    if len(q) < 2 or len(q) != len(p) or len(q) & (len(q) - 1):
        raise ValueError('Equal power-of-two laws required')
    if min(q) < 0 or sum(q) != 1:
        raise ValueError('Normalized program law required')
    if any(not 0 <= a <= b <= 1 for a, b in p):
        raise ValueError('Physical probability intervals required')
    if not sum(a for a, b in p) <= 1 <= sum(b for a, b in p):
        raise ValueError('Target intervals exclude unit mass')
    return q, p


def sparse_events(program, target, budgets=(1, 4, 16, 64, 256, 1024, 4096)):
    """Best sum of certified same-side atom excesses under each size budget.

    This extracts certificates from an already computed complete law. It does
    not claim optimality over unknown exact probabilities or arbitrary events.
    """
    q, p = checked_laws(program, target)
    if any(type(k) is not int or k < 1 for k in budgets):
        raise ValueError('Positive integer budgets required')
    rows = []
    for side in ('program_excess', 'target_excess'):
        scores = [x-b if side == 'program_excess' else a-x for x, (a, b) in zip(q, p)]
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        for budget in (*budgets, 'all'):
            masks = ranked if budget == 'all' else ranked[:budget]
            Q = sum((q[i] for i in masks), F(0))
            P = (sum((p[i][0] for i in masks), F(0)), sum((p[i][1] for i in masks), F(0)))
            lower = max(F(0), Q-P[1] if side == 'program_excess' else P[0]-Q)
            rows.append(dict(side=side, budget=budget, masks=masks, sampler_probability=Q,
                             target_probability_interval=P, TV_lower=lower))
    return rows


def split_gain_interval(x, y):
    """Range of (|x|+|y|-|x+y|)/2 over two real intervals."""
    a, b = map(F, x)
    c, d = map(F, y)
    if a > b or c > d:
        raise ValueError('Ordered intervals required')
    return max(F(0), min(-b, c), min(a, -d)), max(F(0), min(-a, d), min(b, -c))


def prefix_split_witness(program, target):
    q, p = checked_laws(program, target)
    half = len(q) // 2
    rows = []
    fine_mid = coarse_mid = F(0)
    for i in range(half):
        x = (q[i]-p[i][1], q[i]-p[i][0])
        y = (q[i+half]-p[i+half][1], q[i+half]-p[i+half][0])
        a, b = sum(x)/2, sum(y)/2
        fine_mid += (abs(a)+abs(b))/2
        coarse_mid += abs(a+b)/2
        gain = (abs(a)+abs(b)-abs(a+b))/2
        lower, upper = split_gain_interval(x, y)
        assert lower <= gain <= upper
        orientation = 1 if x[1] < 0 < y[0] else -1 if y[1] < 0 < x[0] else 0
        rows.append(dict(parent=i, child_difference_intervals=(x, y), gain_interval=(lower, upper),
                         midpoint_gain=gain, certified_opposite_orientation=orientation))
    total = (sum(r['gain_interval'][0] for r in rows), sum(r['gain_interval'][1] for r in rows))
    assert sum(r['midpoint_gain'] for r in rows) == fine_mid-coarse_mid
    return dict(rows=rows, gain_interval=total, fine_midpoint_half_l1=fine_mid,
                coarse_midpoint_half_l1=coarse_mid, midpoint_gain=fine_mid-coarse_mid,
                certified_opposite_parents=sum(r['certified_opposite_orientation'] != 0 for r in rows))
