"""Positive endpoint fixtures whose queried marginals fit even binary64."""
import numpy as np
from gbslab.positive_completion import EventEstimate


class ConditionalIndependentEvents:
    """Five features conditionally independent given the last, balanced bit.

    Every actually queried singleton/pair event is representable; the product
    of five likelihood ratios can reach exp(+-800). No pre-underflowed table.
    """
    m=6
    def __init__(self,log_odds):
        low=np.exp(-np.longdouble(abs(log_odds))/5)/2
        self.probabilities=(low,np.longdouble(.5)) if log_odds>=0 else (np.longdouble(.5),low)
        self.queried=[]

    def event_probability(self,zeros,ones):
        total=np.longdouble(0)
        for label in (0,1):
            if (5 in zeros and label) or (5 in ones and not label): continue
            value=np.longdouble(.5); p=self.probabilities[label]
            for j in range(5):
                if j in zeros: value*=1-p
                elif j in ones: value*=p
            total+=value
        assert total>0 and np.isfinite(total), 'Fixture event must remain strictly positive and representable'
        self.queried.append((tuple(zeros),tuple(ones),total))
        return EventEstimate(total,provenance='analytic_class_conditional_independence')
