"""Target-interval diagnostics for a fixed, already chosen equal mixture."""
from fractions import Fraction as F
from .interval_completion import physical_tv_interval


def cancellation_geometry(p,q,target):
    p=tuple(map(F,p));q=tuple(map(F,q));target=tuple((F(a),F(b)) for a,b in target)
    if not p or len(p)!=len(q) or len(p)!=len(target) or min(p+q)<0 or sum(p)!=1 or sum(q)!=1 or any(not 0<=a<=b<=1 for a,b in target) or not sum(a for a,b in target)<=1<=sum(b for a,b in target):raise ValueError('Normalized laws and feasible atom intervals required')
    bins={};signs=dict(shared_over=0,shared_under=0,opposite=0,exact_zero=0,unresolved=0);G=[F(0),F(0)];common_over=[F(0),F(0)];common_under=[F(0),F(0)]
    for mask,(a,b,(l,h)) in enumerate(zip(p,q,target)):
        points=[l,h]+[v for v in (a,b,(a+b)/2) if l<=v<=h]
        gains=[(abs(a-t)+abs(b-t)-abs(a+b-2*t))/4 for t in points];over=[min(max(F(0),a-t),max(F(0),b-t)) for t in points];under=[min(max(F(0),t-a),max(F(0),t-b)) for t in points];values=[(min(gains),max(gains)),(min(over),max(over)),(min(under),max(under))];row=bins.setdefault(mask.bit_count(),dict(atoms=0,cancellation=[F(0),F(0)],shared_over=[F(0),F(0)],shared_under=[F(0),F(0)]));row['atoms']+=1
        for name,total,value in zip(('cancellation','shared_over','shared_under'),(G,common_over,common_under),values):
            for k in (0,1):row[name][k]+=value[k];total[k]+=value[k]
        label='exact_zero' if a==b==l==h else 'shared_over' if min(a,b)>h else 'shared_under' if max(a,b)<l else 'opposite' if min(a,b)<l<=h<max(a,b) else 'unresolved';signs[label]+=1
    ptv=physical_tv_interval(p,target);qtv=physical_tv_interval(q,target);mtv=physical_tv_interval([(a+b)/2 for a,b in zip(p,q)],target);diff=(ptv[0]-qtv[1],ptv[1]-qtv[0]);imbalance=(F(0) if diff[0]<=0<=diff[1] else min(abs(x) for x in diff)/2,max(abs(x) for x in diff)/2);margin=(G[0]-imbalance[1],G[1]-imbalance[0]);order=sum(abs(a-b) for a,b in zip(p,q))/2
    assert 0<=G[0]<=G[1]<=order/2
    return dict(forward_TV=ptv,reverse_TV=qtv,mixture_TV=mtv,branch_imbalance_interval=imbalance,cancellation_interval=tuple(G),improvement_over_best_interval=margin,order_TV=order,shared_over_interval=tuple(common_over),shared_under_interval=tuple(common_under),sign_counts=signs,click_bins=bins,scope='Post-hoc explanation of fixed mixture; no fitting weights or attributing native recursion/clipping causes')
