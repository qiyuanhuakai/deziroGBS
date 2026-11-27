#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Each port contributes two adjacent quadratures. Reuse one Schur complement
// for the whole subtree of masks that contain that port.
static void visit(const Complex* matrix, int stride, int sites, int bit, int mask,
                  Complex amplitude, Complex* out, Complex* scratch) {
    if (sites == 0) {
        out[mask] = amplitude;
        return;
    }
    const int rest = 2 * (sites - 1);
    const Complex* tail = rest ? matrix + 2 * stride + 2 : matrix;
    if (rest) visit(tail, stride, sites - 1, bit * 2, mask, amplitude, out, scratch);
    else out[mask] = amplitude;
    const Complex a=matrix[0], b=matrix[1], d=matrix[stride+1];
    const Complex det=a*d-b*b;
    const Complex next=amplitude/std::sqrt(det);
    if (!rest) {
        out[mask+bit]=next;
        return;
    }
    for (int i=0;i<rest;++i) {
        const Complex x=matrix[2+i], y=matrix[stride+2+i];
        const Complex u=(d*x-b*y)/det, v=(a*y-b*x)/det;
        for (int j=i;j<rest;++j) {
            const Complex value=tail[i*stride+j]-u*matrix[2+j]-v*matrix[stride+2+j];
            scratch[i*rest+j]=value;
            scratch[j*rest+i]=value;
        }
    }
    visit(scratch,rest,sites-1,2*bit,mask+bit,next,out,scratch+rest*rest);
}

extern "C" void panel_void118(const Complex* matrices, int nodes, int sites,
                              int threads, Complex* out) {
    const int width=2*sites;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int k=0;k<nodes;++k) {
        std::vector<Complex> scratch(width*width*(sites+1));
        visit(matrices+k*width*width,width,sites,1,0,1.,out+k*(1<<sites),scratch.data());
    }
}
