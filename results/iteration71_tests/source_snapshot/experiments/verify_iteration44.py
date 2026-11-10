"""Independent physical construction, rational elimination, bridge and native laws."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from verify_iteration43_completion_recovery import direct_log
from iteration39_independent import replay
from iteration41_independent import layer_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration44_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration44','iteration22_external','iteration33_run','verify_iteration34','verify_iteration43_completion_recovery','iteration39_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};checks=[];deps={};rows=[];detcache={};detcount=0;steps=0
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    for folder,count in (('panel',24),('native',48)):
        directory='results/iteration44_'+folder;plan=read(directory+'/plan.json');summary=read(directory+'/summary.json');check(folder+' complete',summary['planned']==summary['finished']==summary['completed']==count and summary['unchanged_sources'])
        for n,s in plan['source_hashes'].items():check(folder+' source '+n,digest(ROOT/n)==s==digest(ROOT/directory/'source_snapshot'/n))
        for n,s in plan['dependencies'].items():check(folder+' dependency '+n,digest(ROOT/n)==s);deps[n]=s
    plan=read('results/iteration44_panel/plan.json')
    for index,job in enumerate(plan['jobs']):
        rec=read(f'results/iteration44_panel/case_{index:02d}.json');tag=f'case{index} ';n=job['modes'];size=1<<n;p=rec['physical'];V=p['covariance'];order=job['order'];eta=job['transmission'];check(tag+'job',rec['job']==job and p['order']==order and p['transmission']==eta and sorted(order)==list(range(n)))
        phase=([1,1j,-1,1] if n==4 else [1,1j,-1,1,1j,-1j,1,-1]);A=[[sum(F(int(complex(phase[k]).real))*(-1)**(((i&k).bit_count()+(k&j).bit_count())) for k in range(n))/n for j in range(n)] for i in range(n)];B=[[sum(F(int(complex(phase[k]).imag))*(-1)**(((i&k).bit_count()+(k&j).bit_count())) for k in range(n))/n for j in range(n)] for i in range(n)];check(tag+'fixed interferometer',A==p['unitary_real'] and B==p['unitary_imag']);S=[[A[i][j] if i<n and j<n else -B[i][j-n] if i<n else B[i-n][j] if j<n else A[i-n][j-n] for j in range(2*n)] for i in range(2*n)]
        S=np.array(S,dtype=object);I=np.eye(n,dtype=object);Z=np.zeros((n,n),dtype=object);omega=np.block([[Z,I],[-I,Z]]);check(tag+'exact passive symplectic',np.array_equal(S@S.T,np.eye(2*n,dtype=object)) and np.array_equal(S@omega@S.T,omega))
        if job['strength']=='moderate':q=[F(1,2),F(3,4),F(1,2),F(3,4)]*(n//4);pq=[F(5,2),F(3,2),F(3),F(2)]*(n//4)
        else:q=[F(1,4),F(1,2),F(1,4),F(1,2)]*(n//4);pq=[F(5),F(3),F(6),F(5,2)]*(n//4)
        check(tag+'physical squeezed thermal input',q==p['input_q'] and pq==p['input_p'] and 0<eta<=1 and all(x>0 and y>0 and x*y>1 for x,y in zip(q,pq)));D=q+pq;ix=order+[n+j for j in order];expected=[[eta*sum(S[a,k]*D[k]*S[b,k] for k in range(2*n))+(1-eta)*int(a==b) for b in ix] for a in ix];check(tag+'exact loss covariance',expected==V and all(F(float(x))==x for row in V for x in row));w=p['nonclassical_witness'];norm=sum(x*x for x in w);rayleigh=sum(w[a]*(V[a][b]-int(a==b))*w[b] for a in range(2*n) for b in range(2*n));check(tag+'nonclassical quadratic witness',norm==1 and rayleigh==p['nonclassical_rayleigh']<0)
        target=rec['target'];vac=target['vacuum_intervals'];check(tag+'dimensions',target['bits']==192 and len(vac)==size and len(target['determinants'])==size-1 and vac[0]==[F(1),F(1)])
        for mask in range(1,size):
            modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes];Q=tuple(tuple((V[a][b]+int(a==b))/2 for b in ix) for a in ix)
            if Q not in detcache:detcache[Q]=exact_determinant(Q)
            d=detcache[Q];detcount+=1;a,b=vac[mask];check(tag+'independent determinant '+str(mask),d==target['determinants'][mask-1]>=1);check(tag+'root inequalities '+str(mask),0<=a<=b<=1 and a*a*d<=1<=b*b*d and b-a<=F(1,1<<192))
        transformed=[tuple(x) for x in vac]
        for j in range(n):
            for mask in range(size):
                if not(mask>>j)&1:a,b=transformed[mask];c,d=transformed[mask|(1<<j)];transformed[mask]=(a-d,b-c)
        physical=[]
        for mask in range(size):
            a,b=transformed[(size-1)^mask];physical.append((max(F(0),a),min(F(1),b)));check(tag+'target atom '+str(mask),tuple(target['raw_intervals'][mask])==(a,b) and tuple(target['target_intervals'][mask])==physical[-1])
        check(tag+'physical probability consistency',sum(a for a,b in physical)<=1<=sum(b for a,b in physical) and all(0<=a<=b<=1 for a,b in physical))
        subsets=[(j,) for j in range(n)]+[(i,j) for j in range(n) for i in range(j)];X=[[math.prod(1-2*((mask>>j)&1) for j in subset) for subset in subsets] for mask in range(size)];mi=[]
        for subset in subsets:
            if len(subset)==1:a,b=vac[1<<subset[0]];mi.append((2*a-1,2*b-1))
            else:i,j=subset;a,b=vac[(1<<i)|(1<<j)];c,d=vac[1<<i];e,f=vac[1<<j];mi.append((1+4*a-2*d-2*f,1+4*b-2*c-2*e))
        midpoint=[(a+b)/2 for a,b in mi];check(tag+'physical moment intervals',encode(mi)==encode(target['moment_intervals']) and midpoint==rec['moment_midpoints']);projection=rec['projection'];R=projection['law'];check(tag+'positive normalized completion',len(R)==size and min(R)>0 and sum(R)==1)
        for j,mu in enumerate(midpoint):check(tag+'midpoint moment '+str(j),sum(R[k]*X[k][j] for k in range(size))==mu)
        weights=rec['fit']['probabilities'];qnorm=[x/sum(weights) for x in weights];delta=[mu-sum(qnorm[k]*X[k][j] for k in range(size)) for j,mu in enumerate(midpoint)];check(tag+'exact projection',qnorm==projection['normalized_candidate'] and delta==projection['corrections'] and all(R[k]==qnorm[k]+sum(d*x for d,x in zip(delta,X[k]))/size for k in range(size)))
        bridge=rec['bridge'];radii=[(b-a)/2 for a,b in mi];rho=sum(radii)/size;tvbridge=sum(radii)/2;theta=rec['fit']['theta'];check(tag+'bridge inputs',bridge['rational_moments']==midpoint and encode(mi)==encode(bridge['moment_intervals']) and radii==bridge['moment_radii'] and rho==bridge['atom_radius'] and tvbridge==bridge['TV_to_true_feasible_upper'] and theta==bridge['theta'] and min(R)>rho);intervals=[(x-rho,x+rho) for x in R];check(tag+'positive true-feasible enclosure',encode(intervals)==encode(bridge['feasible_atom_intervals']))
        residual=[]
        for k,(a,b) in enumerate(intervals):
            alo,ahi=direct_log(a);blo,bhi=direct_log(b);l,h=bridge['log_intervals'][k];check(tag+'independent bridge log '+str(k),l<=alo<=ahi<=blo<=bhi<=h);energy=sum(t*x for t,x in zip(theta,X[k]));residual.append((l-energy,h-energy))
        floor=min(l for l,h in residual);gap=sum(b*(h-floor) for (a,b),(l,h) in zip(intervals,residual));eps=bridge['TV_to_true_maxent_upper'];check(tag+'robust entropy upper',floor==bridge['residual_floor'] and gap==bridge['entropy_gap_upper_for_true_feasible']>=0);root=eps-tvbridge;check(tag+'bridge plus Pinsker',eps==1 if gap>=2 else root>=0 and root*root>=gap/2 and (root-F(1,1<<80))**2<gap/2)
        tree=rec['tree'];singles=[((1+midpoint[j])/2,(1-midpoint[j])/2) for j in range(n)];check(tag+'tree singles',encode(singles)==encode(tree['singles']));edges=[(i,j) for i,j,score in tree['selected_edges']];degree=[sum(i==k or j==k for i,j in edges) for k in range(n)];parents=tree['parent'];check(tag+'tree connected',len(edges)==n-1 and sorted(tree['order'])==list(range(n)) and parents[0]==-1 and all(parents[j] in tree['order'][:tree['order'].index(j)] for j in range(1,n)))
        pairs={}
        for offset,(i,j) in enumerate(subsets[n:]):
            values=tuple((1+(-1)**a*midpoint[i]+(-1)**b*midpoint[j]+(-1)**(a+b)*midpoint[n+offset])/4 for a in range(2) for b in range(2));pairs[i,j]=values;check(tag+f'pair {i},{j}',list(values)==tree['pairs'][str((i,j))] and min(values)>0)
        independently=[]
        for mask in range(size):
            numerator=math.prod(pairs[i,j][2*((mask>>i)&1)+((mask>>j)&1)] for i,j in edges);denominator=math.prod(singles[j][(mask>>j)&1]**(degree[j]-1) for j in range(n));independently.append(numerator/denominator)
        T=tree['law'];check(tag+'independent undirected tree law',T==independently and min(T)>0 and sum(T)==1)
        for i,j in edges:
            for a in range(2):
                for b in range(2):check(tag+f'preserved edge {i},{j}:{a}{b}',sum(T[k] for k in range(size) if ((k>>i)&1)==a and ((k>>j)&1)==b)==pairs[i,j][2*a+b])
        for child in range(1,n):
            par=parents[child];i,j=sorted((child,par));expected=tuple((pairs[i,j][2*a+b] if par==i else pairs[i,j][2*b+a])/singles[par][a] for a in range(2) for b in range(2));check(tag+'conditional '+str(child),list(expected)==tree['conditional'][str(child)] and sum(expected[:2])==sum(expected[2:])==1)
        def tv(law,target):return (sum(max(F(0),a-x,x-b) for x,(a,b) in zip(law,target))/2,min(F(1),sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(law,target))/2))
        rtv=tv(R,physical);ttv=tv(T,physical);mtv=(max(F(0),rtv[0]-eps),min(F(1),rtv[1]+eps));check(tag+'separate physical errors',rtv==tuple(rec['physical_TV_interval']) and ttv==tuple(rec['tree_TV_interval']) and mtv==tuple(rec['true_maxent_physical_TV_interval']));row=dict(case=index,modes=n,completion_TV=rtv,tree_TV=ttv,true_maxent_physical_TV=mtv,maxent_distance_upper=eps,true_moment_bridge_upper=tvbridge)
        for cutoff in (2,4):
            label=f'case_{index:02d}_k{cutoff}';directory='results/iteration44_native';raw=read(directory+'/'+label+'.stdout.json');reverse=read(directory+'/'+label+'.reverse.json');audit=read(directory+'/'+label+'.audit.json');check(tag+f'K{cutoff} identity',audit['job']==dict(case=index,cutoff=cutoff,seed=441000+index) and raw['modes']==10 and raw['cutoff']==cutoff and raw['seed']==441000+index and raw['samples']==3);table={x['prefix']:x['threshold_hex'] for x in raw['prefix_tree']};by_prefix={x['prefix']:x for x in raw['prefix_tree']};check(tag+f'K{cutoff} complete/reverse',len(table)==1023 and by_prefix=={x['prefix']:x for x in reverse['prefix_tree']} and all(''.join(str((mask>>j)&1) for j in range(depth)) in table for depth in range(10) for mask in range(1<<depth)));steps+=replay(raw)
            csvname=directory+'/'+label+'.csv';deps[csvname]=digest(ROOT/csvname);stored=np.loadtxt(ROOT/csvname,delimiter=',');ix=list(range(n))+list(range(10,10+n));check(tag+f'K{cutoff} exact covariance',all(F(float(stored[i,j]))==(V[ix.index(i)][ix.index(j)] if i in ix and j in ix else F(i==j)) for i in range(20) for j in range(20)))
            for prefix,x in by_prefix.items():
                q,prev,z=(float.fromhex(x[k]) for k in ('threshold_hex','previous_hex','proposed_hex'));check(tag+f'K{cutoff} quotient '+prefix,math.isnan(q) and prev==z==0 if prev==0 else q==float(F(z)/F(prev)))
                if prefix:
                    par=by_prefix[prefix[:-1]];pp=float.fromhex(par['previous_hex']);pz=float.fromhex(par['proposed_hex']);check(tag+f'K{cutoff} continuity '+prefix,prev==(pz if prefix[-1]=='0' else float(F(pp)-F(pz))))
            padded=physical+[(F(0),F(0))]*(1024-size)
            for model in ('continuous','uint64'):
                law=layer_law(table,10,model);check(tag+f'K{cutoff} independent {model}',law==audit['laws'][model]['distribution'] and min(law)>=0 and sum(law)==1);value=tv(law,padded);check(tag+f'K{cutoff} target TV {model}',value==tuple(audit['physical_TV_intervals'][model]) and sum(law[size:])==audit['vacuum_padding_leak'][model])
                if model=='uint64':row[f'K{cutoff}_TV']=value;row[f'completion_better_K{cutoff}']=rtv[1]<value[0];row[f'completion_worse_K{cutoff}']=rtv[0]>value[1];row[f'true_maxent_worse_K{cutoff}']=mtv[0]>value[1]
        rows.append(row);print('verified case',index,'R',float(rtv[0]),'K2',float(row['K2_TV'][0]),'K4',float(row['K4_TV'][0]),flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=24,native_configurations=48,exact_determinant_checks=detcount,distinct_exact_determinants=len(detcache),replayed_steps=steps,source_hashes=hashes,dependencies=deps,completion_better_K2=sum(r['completion_better_K2'] for r in rows),completion_worse_K2=sum(r['completion_worse_K2'] for r in rows),completion_worse_K4=sum(r['completion_worse_K4'] for r in rows),true_maxent_worse_K2=sum(r['true_maxent_worse_K2'] for r in rows)));print('verified total',len(checks),flush=True)
if __name__=='__main__':main()
