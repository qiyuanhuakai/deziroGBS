"""Independent analytic photon laws validate the bivariate reference machinery."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import mpmath as mp
spec=importlib.util.spec_from_file_location('optical86',Path(__file__).resolve().parents[1]/'experiments/iteration86_optical_reference_v1.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
spec2=importlib.util.spec_from_file_location('optical86v2',Path(__file__).resolve().parents[1]/'experiments/iteration86_optical_reference_v2.py')
general=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(general)

class OpticalReference86Tests(unittest.TestCase):
 def setUp(self):mp.mp.dps=80
 def test_independent_thermal_geometric_product(self):
  a,b=.25,.5;x=np.diag([a,b,a,b]);poly=ref.determinant_polynomial(x);p=ref.coefficients(poly,12)
  for i in range(13):
   for j in range(13):self.assertLess(abs(p[i][j]-(mp.mpf(a)/(1+a))**i/(1+a)*(mp.mpf(b)/(1+b))**j/(1+b)),mp.mpf('1e-65'))
  self.assertLessEqual(1-sum(map(sum,p)),sum(ref.tail_bound([x],j,12) for j in range(2)))
 def test_two_mode_squeezed_equal_photon_numbers(self):
  # Exact dyadic rational covariance: n=9/16 and sqrt(n(n+1))=15/16.
  n=mp.mpf(9)/16;c=mp.mpf(15)/16;x=np.array([[n,c,0,0],[c,n,0,0],[0,0,n,-c],[0,0,-c,n]],float);p=ref.coefficients(ref.determinant_polynomial(x),12)
  for i in range(13):
   for j in range(13):
    expected=(n/(1+n))**i/(1+n) if i==j else 0;self.assertLess(abs(p[i][j]-expected),mp.mpf('1e-60'))
 def test_two_spectra_convolution_and_fourier(self):
  xs=[np.diag([.25,.5,.25,.5]),np.diag([.125,.25,.125,.25])];laws=[ref.coefficients(ref.determinant_polynomial(x),10) for x in xs];combined=ref.coefficients(ref.multiply(*[ref.determinant_polynomial(x) for x in xs]),10)
  for i in range(11):
   for j in range(11):
    value=sum(laws[0][k][l]*laws[1][i-k][j-l] for k in range(i+1) for l in range(j+1));self.assertLess(abs(combined[i][j]-value),mp.mpf('1e-65'))
  fft=ref.fourier(xs,64);self.assertLess(np.max(abs(fft[:11,:11]-np.array(combined,float))),1e-13)
 def test_full_bin_reference_retains_multiple_occupied_modes(self):
  # Two independent thermal modes in each aggregate: exact negative binomial.
  x=np.eye(8)*.25;law=general.fourier_general([x],[0,0,1,1],64).real
  for i in range(12):
   for j in range(12):self.assertAlmostEqual(law[i,j],(i+1)*(j+1)*.2**(i+j)*.8**4,places=13)
  marginal,tail=general.marginal_reference([x],[0,1],12)
  self.assertLessEqual(1-sum(marginal),tail)

if __name__=='__main__':unittest.main()
