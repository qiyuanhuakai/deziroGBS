"""Create a separate warp-reduction variant; preserve fused v1 evidence."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'native/gaussian_click_fused74_v1.cu').read_text()
old='''            if(t==0){double av=0,bv=0,ab=0,ba=0,mq=0,mp=0,c[6]={};for(int i=0;i<d;++i){add72(a[i]*xa[i],av,c[0]);add72(b[i]*xb[i],bv,c[1]);add72(a[i]*xb[i],ab,c[2]);add72(b[i]*xa[i],ba,c[3]);add72(a[i]*W[k*d+i],mq,c[4]);add72(b[i]*W[k*d+i],mp,c[5]);}
'''
new='''            // Fixed four-warp tree; the original serial compensated sums
            // remain in v1. This changes rounding and needs probability checks.
            double terms[6]={},corr[6]={};
            for(int i=t;i<d;i+=blockDim.x){add72(a[i]*xa[i],terms[0],corr[0]);add72(b[i]*xb[i],terms[1],corr[1]);add72(a[i]*xb[i],terms[2],corr[2]);add72(b[i]*xa[i],terms[3],corr[3]);add72(a[i]*W[k*d+i],terms[4],corr[4]);add72(b[i]*W[k*d+i],terms[5],corr[5]);}
            for(int shift=16;shift;shift/=2)for(int j=0;j<6;++j)terms[j]+=__shfl_down_sync(0xffffffff,terms[j],shift);
            if(t%32==0)for(int j=0;j<6;++j)partials[j*4+t/32]=terms[j];__syncthreads();
            if(t==0){double sums[6]={},c[6]={};for(int w=0;w<4;++w)for(int j=0;j<6;++j)add72(partials[j*4+w],sums[j],c[j]);double av=sums[0],bv=sums[1],ab=sums[2],ba=sums[3],mq=sums[4],mp=sums[5];
'''
assert source.count(old)==1
source=source.replace(old,new).replace('__shared__ int n,clicked','__shared__ double partials[24];\n    __shared__ int n,clicked')
rng=(ROOT/'native/gaussian_click_philox74_v1.cu').read_text().replace('gaussian_click_fused74_v1.cu','gaussian_click_fused74_v2.cu')
pilot=(ROOT/'experiments/iteration74_fused_pilot_v1.py').read_text().replace('gaussian_click_philox74_v1.cu','gaussian_click_philox74_v2.cu').replace('gaussian_click_fused74_v1.cu','gaussian_click_fused74_v2.cu').replace('Build fused-overflow','Build fused-overflow warp-reduction')
comparison=(ROOT/'experiments/iteration74_fused_comparison_v1.py').read_text().replace('results/iteration74_fused_pilot_v1/philox.dll','results/iteration74_fused_pilot_v2/philox.dll')
for name,data in [('native/gaussian_click_fused74_v2.cu',source),('native/gaussian_click_philox74_v2.cu',rng),('experiments/iteration74_fused_pilot_v2.py',pilot),('experiments/iteration74_fused_comparison_v2.py',comparison)]:
    with (ROOT/name).open('xb') as f:f.write(data.encode())
print('Created v2 warp reduction sources.')
