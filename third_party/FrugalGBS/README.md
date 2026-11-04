FrugalGBS
=========

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=arXiv:2511.14923&color=active&style=flat-square)](https://arxiv.org/abs/2511.14923)

Summary
-------

A C++/CUDA library for cumulant computation, double-elision sampling, and bitstring statistical analysis.

Derived from the code used for the results of ["A Fast and Frugal Gaussian Boson Sampling Emulator"](https://arxiv.org/abs/2511.14923).

Locally install FrugalGBS for CPU project
-----------------------------------------

1. Clone and locally install Eigen:

```bash
git clone -b 3.4 https://github.com/PX4/eigen.git
cmake -S eigen -B build/eigen \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local/lib"
cmake --install build/eigen
```

2. Clone and locally install xsimd:

```bash
git clone https://github.com/xtensor-stack/xsimd.git
cmake -S xsimd -B build/xsimd \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local/lib"
cmake --install build/xsimd
```

3. Clone, configure, build, and locally install FrugalGBS:

```bash
git clone https://github.com/tomdodd4598/FrugalGBS.git
cmake -S FrugalGBS -B build/fgbs \
  -DBUILD_TESTING=OFF \
  -DFGBS_ENABLE_CUDA=OFF \
  -DFGBS_EIGEN3_DIR="$PWD/.local/lib/share/eigen3/cmake" \
  -DFGBS_XSIMD_DIR="$PWD/.local/lib/share/cmake/xsimd" \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local/lib"
cmake --build build/fgbs -j
cmake --install build/fgbs
```

4. Configure `CMAKE_PREFIX_PATH` to be the local install directory:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/.local/lib"
```

5. In your project's `CMakeLists.txt`, find and link FrugalGBS:

```cmake
find_package(FrugalGBS CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE FrugalGBS::FrugalGBS_CPU)
```

Locally set up FrugalGBS for CPU build
--------------------------------------

1. Clone and locally install Eigen:

```bash
git clone -b 3.4 https://github.com/PX4/eigen.git
cmake -S eigen -B build/eigen \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local/eigen"
cmake --install build/eigen
```

2. Clone and locally install xsimd:

```bash
git clone https://github.com/xtensor-stack/xsimd.git
cmake -S xsimd -B build/xsimd
cmake --install build/xsimd
```

3. Clone and locally install Catch2:

```bash
git clone https://github.com/catchorg/Catch2.git
cmake -S Catch2 -B build/catch2 \
  -DBUILD_TESTING=OFF \
  -DCATCH_BUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/.local/catch2"
cmake --build build/catch2 -j
cmake --install build/catch2
```

4. Configure and build FrugalGBS:

```bash
cmake -S . -B build/fgbs \
  -DBUILD_TESTING=ON \
  -DFGBS_ENABLE_CUDA=OFF \
  -DFGBS_EIGEN3_DIR="$PWD/.local/eigen/share/eigen3/cmake" \
  -DFGBS_XSIMD_DIR="$PWD/build/xsimd" \
  -DFGBS_CATCH2_DIR="$PWD/.local/catch2/lib/cmake/Catch2"
cmake --build build/fgbs -j
```

5. Run the FrugalGBS test suite:

```bash
ctest --test-dir build/fgbs --output-on-failure
```
