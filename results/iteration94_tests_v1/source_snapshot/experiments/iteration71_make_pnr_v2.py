"""Versioned source transformation; preserves already-executed v1 verbatim."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'native/gaussian_pnr_branches_v1.cpp').read_text(encoding='utf-8')
helper=r'''
// Analytically merge each component's click moments before multiplying the
// signed mixture weights. This avoids differencing nearly equal covariances.
// The condition limit is an explicit approximation control, not clipping.
static void stable_jump72(std::vector<Branch71>& branches,const Step71& step,int budget,double condition_limit,Trace71& trace){
    need(step.p1>0,"Zero finite-tap herald probability");
    long double predicted=0;for(std::size_t k=0;k<branches.size();++k)predicted+=std::abs(branches[k].weight)*(1+step.parts[k].p0)/step.p1;
    if(branches.size()*2<=std::size_t(budget)&&predicted<=condition_limit){update71(branches,step,true,budget,trace);return;}
    std::vector<Branch71> moments;for(std::size_t k=0;k<branches.size();++k){auto& p=step.parts[k];if(p.p1==0)continue;auto b=branches[k];click_update70(b.x,b.w,p,true);b.weight*=p.p1/step.p1;moments.push_back(std::move(b));}
    need(!moments.empty(),"Empty finite-tap conditional moments");auto merged=merge71(moments);branches.clear();branches.push_back(std::move(merged));++trace.merges;
}
'''
source=source.replace('// Experimental PNR trajectories','// Version 2: condition-aware analytic moment compression.\n// Experimental PNR trajectories',1)
source=source.replace('// Probe finite-tap conditional',helper+'\n// Probe finite-tap conditional',1)
source=source.replace('API int pnr71_sample(', 'API int pnr72_sample(',1)
source=source.replace('int budget,double epsilon,const double* mean','int budget,double epsilon,double condition_limit,const double* mean',1)
source=source.replace('&&epsilon>0&&epsilon<1&&mean','&&epsilon>0&&epsilon<1&&std::isfinite(condition_limit)&&condition_limit>=1&&mean',1)
old='update71(branches,step,true,budget,trace);'
assert source.count(old)==2
position=source.index('API int pnr72_sample(')
source=source[:position]+source[position:].replace(old,'stable_jump72(branches,step,budget,condition_limit,trace);',1)
destination=ROOT/'native/gaussian_pnr_branches_v2.cpp'
with destination.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
print(destination)
