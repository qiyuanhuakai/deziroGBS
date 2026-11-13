The raw data files are some “*.bin” files for a large number of samples. Each file includes ~1.5 million samples. Each sample is stored as 144 bytes, representing 1152 click modes. Each 8 bit in one byte is sampled from one fiber-loop.

Example (Python code):
```python
data = np.unpackbits(
    np.fromfile(
        "*.bin",
        dtype=np.uint8
    )
).reshape(-1, 1152)
```

This python code import one data file as n * m matrix in the dtype of uint8, where n is the number of samples and m is the number of modes(=1152). Each of the 8 adjacent cols([0..7], [8..15],...) is from the same fiber-loop.


The squeezing parameter is stored in "*.xlsx". The 2 cols are two modes for the partial photon indistinguishability. The 25 rows are 25 indepedent TMSV sources.


The transformation matrix is “*.mtx”, which is stored in Matrix Market Array Format (detailed description see https://math.nist.gov/MatrixMarket/formats.html and https://reference.wolfram.com/language/ref/format/MTX.html).
The rows of the matrix are arranged according to {1H, 1V, 2H, 2V, ..., 25H, 25V}. It needs to be rearranged to {1H, 2H, ..., 25H, 1V, 2V, ..., 25V} when calculating the output probability.

