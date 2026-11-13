"""Create, never overwrite, the fused-overflow CUDA experiment sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=(ROOT/'native/gaussian_click_cuda72_v1.cu').read_text()
marker='''        }else{
            // Descending fork preserves old slots until each has been consumed.'''
fused='''        }else if(clicked && 2*n>budget){
            // All 2*n children will immediately be merged. Preserve the same
            // pair ordering/normalization without materializing child matrices.
            if(t==0){
                for(int k=n-1;k>=0;--k){double w=weights[k];weights[2*k]=w/p1;weights[2*k+1]=-w*params[8*k]/p1;}
                double z=0,c=0;for(int k=0;k<2*n;++k)add72(weights[k],z,c);mass=z;
                if(!(isfinite(mass)&&mass>0))err=7;double condition=0;
                for(int k=0;k<2*n;++k){weights[k]/=mass;condition+=fabs(weights[k]);if(!isfinite(weights[k])||weights[k]==0)err=8;}
                maximum_condition=fmax(maximum_condition,condition);peak=max(peak,2*n);
            }
            __syncthreads();if(err){if(t==0)where=mode;break;}
            for(int i=t;i<d;i+=blockDim.x){double value=0,c=0;
                for(int k=0;k<n;++k){double before=W[k*d+i],after=before-V[std::size_t(k)*3*d+2*d+i];
                    add72(weights[2*k]*before,value,c);add72(weights[2*k+1]*after,value,c);}
                average[i]=value;
            }__syncthreads();
            for(std::size_t index=t;index<dd;index+=blockDim.x){int i=index%d,j=index/d;if(i>j)continue;double value=0,c=0;
                for(int k=0;k<n;++k){double* xa=V+std::size_t(k)*3*d,*xb=xa+d,*delta=xb+d,*p=params+8*k;
                    double before=X[std::size_t(k)*dd+index],after=before-u72(i,j,xa,xb,p);
                    double wi=W[k*d+i],wj=W[k*d+j],vi=wi-delta[i],vj=wj-delta[j];
                    add72(weights[2*k]*(before+(wi-average[i])*(wj-average[j])),value,c);
                    add72(weights[2*k+1]*(after+(vi-average[i])*(vj-average[j])),value,c);
                }
                X[index]=X[std::size_t(i)*d+j]=value;if(!isfinite(value))atomicExch(&err,6);
            }
            __syncthreads();for(int i=t;i<d;i+=blockDim.x)W[i]=average[i];
            if(t==0){n=1;weights[0]=1;++merges;}__syncthreads();
        }else{
            // Descending fork preserves old slots until each has been consumed.'''
assert old.count(marker)==1
kernel=old.replace(marker,fused).replace('int slots=2*budget','int slots=budget').replace('slots=std::size_t(count)*2*budget','slots=std::size_t(count)*budget')
rng=(ROOT/'native/gaussian_click_philox73_v1.cu').read_text().replace('gaussian_click_cuda72_v1.cu','gaussian_click_fused74_v1.cu').replace('slots=std::size_t(count)*2*budget','slots=std::size_t(count)*budget')
for name,source in [('gaussian_click_fused74_v1.cu',kernel),('gaussian_click_philox74_v1.cu',rng)]:
    with (ROOT/'native'/name).open('xb') as f:f.write(source.encode())
print('Created fused kernel and Philox entry point; arithmetic order preserved per matrix entry.')
