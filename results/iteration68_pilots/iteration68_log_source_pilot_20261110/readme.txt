1. Filtration of Quantum Channel

Execute JZ4_make_cov.py to compute the mixed covariance matrix from the raw experimental data of the Jiuzhang 4.0 quantum computer. The required datasets are S64 (7nJ) and M256 (7nJ).

Run grad_S_dyn.npy to decompose the mixed bosonic Gaussian state into a combination of pure Gaussian states and random displacements.

2. PCO mapping

Execute cal_GBS_ts.jl to generate all 144 local tensors for the J2-P65-5 configuration.

Use overlap.py to check the overlaps between the MPSs generated using our method and the hafnian-based method.

Note: We currently provide only D=200 results as a preliminary demonstration. You may also generate Matrix Product States (MPS) with larger bond dimensions (D values) using both our method and the traditional Hafnian-based method (code is available via https://zenodo.org/records/7992736) to compare their computational results and efficiency.


3. Time log

contains all the wall time results presented in the paper, which can be used for reproducing and validating the performance data.