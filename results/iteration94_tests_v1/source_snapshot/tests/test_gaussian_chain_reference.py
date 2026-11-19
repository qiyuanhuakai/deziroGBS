import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from iteration24_reference import global_elimination,low_sector_mps,low_sector_gaussian,partial_state_bounds
from gbslab.gaussian_kernel_chain import fold_kernel_chain
from gbslab.mps_overlap import normalized_right_mps
from gbslab.mps_probe import mps_to_dense
from gbslab.banded_gaussian_mps import pairing_log_normalization
from test_banded_gaussian_mps import reference_box


class GaussianChainReferenceTests(unittest.TestCase):
    def test_global_sparse_elimination_and_determinant(self):
        rng=np.random.default_rng(240301)
        for m in (2,4,7):
            bonds=[0]+[2]*(m-1)+[0]; kernels=[]
            for i in range(m):
                n=bonds[i]+1+bonds[i+1]; a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); kernels.append(.03*(a+a.T))
            B,c,_=fold_kernel_chain(kernels,bonds); C,info=global_elimination(kernels,bonds)
            np.testing.assert_allclose(B,C,atol=1e-14,rtol=0)
            self.assertLess(abs(info['log_determinant_abs']+2*c.real),1e-13)
            self.assertLess(abs(complex(*info['determinant_phase'])/np.exp(-2j*c.imag)-1),2e-13)

    def test_low_amplitudes_against_full_tensor(self):
        rng=np.random.default_rng(240302); tensors=[]
        for shape in ((1,4,3),(3,4,3),(3,4,1)): tensors.append(rng.normal(size=shape)+1j*rng.normal(size=shape))
        tensors,_=normalized_right_mps(tensors); dense=mps_to_dense(tensors); low,pairs=low_sector_mps(tensors)
        expected=[dense[(0,0,0)]]
        for i,j in pairs:
            occ=[0]*3; occ[i]+=1; occ[j]+=1; expected.append(dense[tuple(occ)])
        np.testing.assert_allclose(low,expected,atol=1e-14,rtol=0)

    def test_partial_bounds_against_full_gaussian_overlap(self):
        rng=np.random.default_rng(240303); B=np.diag([.18,.25j,.14]); tensors=[rng.normal(size=s)+1j*rng.normal(size=s) for s in ((1,4,2),(2,4,2),(2,4,1))]
        tensors,_=normalized_right_mps(tensors); dense=mps_to_dense(tensors); low,pairs=low_sector_mps(tensors)
        gaussian=low_sector_gaussian(B,pairing_log_normalization(B),pairs); bounds=partial_state_bounds(low,gaussian)
        target=reference_box(B,4); actual=math.sqrt(1-abs(np.vdot(target,dense))**2)
        self.assertLessEqual(bounds['pure_trace_lower_estimate'],actual+1e-13)
        self.assertAlmostEqual(float(np.sum(np.abs(gaussian[1:])**2)),.5*abs(gaussian[0])**2*float(np.linalg.norm(B)**2),delta=1e-14)
        full_tv=.5*(float(np.sum(np.abs(np.abs(dense)**2-np.abs(target)**2)))+1-float(np.linalg.norm(target)**2))
        self.assertLessEqual(bounds['pnr_coarse_tv_lower_estimate'],full_tv+1e-13)


if __name__=='__main__': unittest.main()
