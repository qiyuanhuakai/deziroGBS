"""Full-output word probabilities and rigorous selected-atom TV bounds."""
from fractions import Fraction as F
from .frugal_law import zero_probability


def path_probability(output, steps, *, model='uint64'):
    if not isinstance(output,str) or not output or set(output)-{'0','1'} or len(steps)!=len(output):
        raise ValueError('A complete binary output and one threshold per step required')
    mass=F(1);factors=[]
    for bit,step in zip(output,steps):
        z=zero_probability(float.fromhex(step[0]),model)
        factor=z if bit=='0' else 1-z;factors.append(factor);mass*=factor
    return dict(probability=mass,factors=factors,model=model)


def selected_overlap(outputs, program, target):
    labels=list(outputs);q=list(map(F,program));p=[tuple(map(F,v)) for v in target]
    if not labels or len(labels)!=len(q) or len(q)!=len(p) or len(set(labels))!=len(labels):
        raise ValueError('Nonempty distinct selected complete outputs required')
    n=len(labels[0])
    if not n or any(len(s)!=n or set(s)-{'0','1'} for s in labels):
        raise ValueError('Equal-length complete binary outputs required')
    if min(q)<0 or sum(q)>1 or any(not 0<=a<=b<=1 for a,b in p) or sum(a for a,b in p)>1:
        raise ValueError('Valid subprobability masses required')
    overlap_lower=sum(min(x,a) for x,(a,b) in zip(q,p))
    overlap_upper=sum(min(x,b) for x,(a,b) in zip(q,p))
    q_rest=1-sum(q);p_rest=(max(F(0),1-sum(b for a,b in p)),1-sum(a for a,b in p))
    upper=min(F(1),1-overlap_lower)
    lower=max(F(0),1-overlap_upper-min(q_rest,p_rest[1]))
    return dict(selected_overlap_interval=(overlap_lower,overlap_upper),program_selected_mass=sum(q),
                target_selected_mass_interval=(sum(a for a,b in p),sum(b for a,b in p)),
                program_remaining_mass=q_rest,target_remaining_mass_interval=p_rest,
                full_TV_interval=(lower,upper),upper_gap_from_one=overlap_lower)
