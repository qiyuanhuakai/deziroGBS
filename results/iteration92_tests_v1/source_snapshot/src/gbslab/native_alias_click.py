"""Small-output direct threshold sampler using integer alias thresholds.
Use gaussian_table_click78_v3.cpp. The low m random-word bits choose a column;
the other 64-m bits choose its outcome. law() exports that ideal-uniform-uint64
law rounded to float64, not the older 53-bit CDF law. Mixer seeds are not iid
certificates. Exponential construction and 16-output limit remain explicit.
"""
from .native_table_click import NativeTableClickSampler
class NativeAliasClickSampler(NativeTableClickSampler):
    """Integer alias version; covariance convention and resource limits as above."""
