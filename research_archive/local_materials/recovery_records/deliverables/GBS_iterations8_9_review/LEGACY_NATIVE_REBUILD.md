# 完整原生回归依赖的跨环境重建

这是恢复说明，不是本轮新增实验。原始 268 项零跳过结果保持在其冻结记录中。当前环境曾使用 Linux x86-64、glibc 2.39、GCC 13.3、AMD EPYC 9V74；`-march=native` 不能视为任意机器可用。所有程序需要兼容的 libstdc++ / libgcc / libc / libm；Frugal 另需 libgomp，新 forest 另需 libquadmath。

## 1. 原始资产和门禁发现路径

恢复包的 `native_cache/` 保留下列八对程序与原始同目录 `build.json`；成员散列均在 MANIFEST.json。

| 相对 .cache 路径 | 用途 |
|---|---|
| iteration8-batched-native/libbatched_forest.so | 新事件内核 |
| iteration8-retained-native/click_sampler | 门禁指定的 retained 基线 |
| iteration8-forest-native/batched_forest_sampler | 新完整原生 CLI |
| click-prefix-native/click_sampler | prefix cache 当前候选 |
| click-iteration4-native/click_sampler | 独立旧版 retained 对照 |
| frugal-native/frugal_sampling_adapter | canonical Frugal 对照 |
| frugal-evidence-native/frugal_evidence_adapter | 冻结 v1 evidence 对照 |
| frugal-evidence-checkpoint-native/frugal_evidence_adapter | checkpoint evidence 实现 |

`verify_iteration8.py` 只主动指定前三项；其余路径由旧测试发现。原门禁同时校验构建记录中 quadmath 运行库与头文件的绝对路径和散列：

- `/usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0`：`353d31394be238ed265989ac09741a2892c14ddabfac0b30f3f26a237a8bd5d2`
- `/usr/lib/gcc/x86_64-linux-gnu/13/include/quadmath.h`：`e8711f597c69a64800a00b80644f66156623323615b03d0cdba4b4c845e44736`

仅复制程序不保证门禁可以在不同环境运行。需要重建时保留原始资产不动，使用下列对应源码，生成新 build.json，再运行新的门禁。不得编辑旧散列绕过验证。

## 2. 当前 retained 和 forest

在从 bundle 恢复的仓库根目录中执行。Eigen 头文件位置需替换为当前机器的实际位置，供应商 Frugal / xsimd 源码已经随仓库提供。

```bash
export EIGEN3_INCLUDE_DIR=/path/to/eigen3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
python experiments/build_click_sampler.py --native --build-dir .cache/iteration8-retained-native
python experiments/build_batched_forest.py --native --build-dir .cache/iteration8-batched-native
python experiments/build_batched_forest_sampler.py --native --build-dir .cache/iteration8-forest-native
export GBSLAB_PREFIX_EXECUTABLE="$PWD/.cache/iteration8-retained-native/click_sampler"
```

最后一个环境变量只指定 prefix 测试的当前候选，不替代旧版 iteration4 对照。本轮保存的 prefix 与 retained 程序字节相同。

## 3. 旧版 retained、canonical Frugal、checkpoint Frugal

从恢复仓库根目录建立独立的历史工作树，避免污染当前源码：

```bash
GBS_RESTORE_ROOT="$PWD"
git worktree add --detach ../gbs-legacy4-restore 37518d58f571de62749c14c322fe7e0736240ed0
cd ../gbs-legacy4-restore
python experiments/build_click_sampler.py --native --build-dir .cache/click-iteration4-native
python experiments/build_frugal_sampling.py --native --modes 8,10,12,16,32,64 --build-dir .cache/frugal-native
python experiments/build_frugal_evidence.py --native --modes 8,10,12,16,32,64,100,144 --build-dir .cache/frugal-evidence-checkpoint-native
mkdir -p "$GBS_RESTORE_ROOT/.cache"
cp -a .cache/click-iteration4-native .cache/frugal-native .cache/frugal-evidence-checkpoint-native "$GBS_RESTORE_ROOT/.cache/"
cd "$GBS_RESTORE_ROOT"
```

此历史提交中的 `click_sampler.cpp` SHA 为 `9f365a5aabb2815406936ae92e7bd0ddd244f021a4b9ebf1f8e4db688ffea7ab`；`build_click_sampler.py` 为 `15eb7d4212299a9418a4b8fd0cbff4434a5c5c0f4161962770dce8c3136d9682`。同提交的 Frugal evidence adapter 已是 checkpoint 版，SHA 为 `85d0009f64c0be66a39a0ff62ac83b1d8a6d3bc504ab6513f10061967f8dbd6d`。

## 4. v1 Frugal evidence 必须来自冻结快照

当前或上述历史工作树中的 checkpoint adapter 不等于 v1。另建一个独立工作树，并把三个 v1 快照放回其编译位置：

```bash
git worktree add --detach ../gbs-frugal-v1-restore 37518d58f571de62749c14c322fe7e0736240ed0
cd ../gbs-frugal-v1-restore
cp results/iteration4_frugal_adapter/source_snapshot/frugal_evidence_adapter.cpp experiments/frugal_evidence_adapter.cpp
cp results/iteration4_frugal_adapter/source_snapshot/build_frugal_evidence.py experiments/build_frugal_evidence.py
cp results/iteration4_frugal_adapter/source_snapshot/build_frugal.py experiments/build_frugal.py
python experiments/build_frugal_evidence.py --native --modes 8,10,12,16,32,64,100,144 --build-dir .cache/frugal-evidence-native
cp -a .cache/frugal-evidence-native "$GBS_RESTORE_ROOT/.cache/"
cd "$GBS_RESTORE_ROOT"
python experiments/verify_iteration8.py --output results/repro8_validation
```

v1 三个快照 SHA256：

| 文件 | SHA256 |
|---|---|
| frugal_evidence_adapter.cpp | 9661b3882885a15f32ddf435bf55ad250978800a70842546566eea897d3b1289 |
| build_frugal_evidence.py | 7ec3a819c574b4b90f1f4e8c34690cdfb362fb00b6b6712a480730305fd876e1 |
| build_frugal.py | d97a6f045d73e8d3ffe59afdcfb1c6fb3848a3da9241bb5a1926958ec8909179 |

这些快照随 bundle 保存。当前 checkpoint builder 保护旧 v1 输出路径，所以仅改变输出参数不能重建独立 v1。以上新生成的程序、build.json 和新验证记录与原归档证据分开保留。
