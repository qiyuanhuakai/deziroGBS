# GBS 第八、九轮完整研究恢复包

日期：2026-11-05（北京时间）。项目：量子 / deziroGBS。
最终提交：`8675497ae2ff16f7f6fbaba4ad6eb0545c1612ff`，分支：`research/iteration8`。
恢复基点：第七轮 `108b732443e273ee1b898dff21b351a14b2a6ce8`。

## 先读结论

第八轮完成受保护的共享 Schur 事件内核和完整原生采样器。新输入的 72 个完整分布、96 项交叉检查、12 个历史回归分布和 108 次原生冷批次运行全部完成。历史 4 行数值失败消除；四种新方法的 TV 比值中位数为 0.726–0.915，但成本为较强 retained 基线的 34.73–40.43 倍，24 个配对计时单元全部更慢。速度目标失败，未将质量改善写成效率突破。

第九轮立即转入纯态核心分解、保留相干性的总光子数截断误差界和表示资源筛查。固定 42 行比较及 20 项检查完成。最轻的 1024 模式核心 K=36 仍有约 10^67 的稠密偶数 Fock 维数，因此排除大规模稠密展开。新的压缩表示采样器尚未实现；分解及 MPS 加位移的整体思想已有先前工作。理论不为现有 forest 采样律提供全局 TV 证书。

主报告位于 `review/RESEARCH_REPORT_ITERATION8.md` 与 `review/RESEARCH_REPORT_ITERATION9.md`。`review/` 提供当前源代码、研究记录、配置、测试和本两轮结果的阅读副本；它不是完整安装目录，历史数据及供应商源码从下面的 bundle 恢复。

## 恢复完整仓库

在本文件所在目录执行：

```bash
git bundle verify history.bundle
git clone -b research/iteration8 history.bundle deziroGBS
cd deziroGBS
git rev-parse HEAD
```

最后一条须输出本页最终提交。`history.bundle` 包含第一至九轮全部已跟踪源代码、供应商源码与许可证、测试、配置、报告、正式数据和历史失败记录。包内 `bundle_verification.txt` 记录自包含性验证；`restore_verification.json` 记录实际在新目录克隆后得到相同提交与干净工作树。原有第一至七轮科学代码和结果未改写。

Python 环境依照恢复后的 `pyproject.toml` / README 安装。Linux 原生构建需要 C++20、Eigen、GNU libquadmath；Frugal 集成另需 OpenMP。归档的 native 构建是在 AMD EPYC 9V74、GCC 13.3 上使用 `-march=native` 产生，不能保证在其他 CPU/系统上直接运行。原生依赖库不随包提供。

## 已验证原生资产

`native_cache/` 显式收录 8 对二进制与其原始 `build.json`，包括第八轮共享库、retained 基线、forest CLI，以及旧集成测试使用的五条路径。它不包括其他计算缓存或原始九章协方差。

在新恢复的仓库中，如果硬件、系统和依赖兼容，可将资产放入对应路径：

```bash
mkdir -p .cache
cp -a ../native_cache/. .cache/
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
python experiments/verify_iteration8.py --output results/repro8_validation
```

此命令从恢复目录 `deziroGBS` 内运行。它重新运行完整门禁并写入新目录；不会编译程序。第八轮冻结记录为 **268 项通过、0 失败、0 错误、0 跳过**，只属于实际验证过的源码、构建和环境。新机器应运行新的验证，不可把已有记录当作移植证明。

若 CPU、运行时或依赖不兼容，按 `review/RESEARCH_REPORT_ITERATION8.md` 第 7 节重编第八轮三项，再建立新的门禁。旧历史回归程序必须保持其对应版本；不能用当前代码重编一个旧文件名，声称恢复了独立旧版基线。精确历史工作树、v1 快照与重建命令见 [LEGACY_NATIVE_REBUILD.md](LEGACY_NATIVE_REBUILD.md)。构建元数据保留原始绝对路径与散列，不应编辑历史文件来绕过校验。

## 运行本两轮实验

完整恢复后，第八轮质量、回归、计时和分析入口见对应主报告第 7 节。所有复跑采用新输出目录，并先建立与实际源码、二进制绑定的 validation。正式成本包含准备与抽样，不把 Python 枚举、解析预算程序或失败后的补抽算作采样加速。

第九轮固定资源计算可独立复跑：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
python experiments/iteration9_core_budget.py freeze --output results/repro9_core_budget
python experiments/iteration9_core_budget.py run --output results/repro9_core_budget
```

原始完整分布、失败质量、R/E 轨迹、计时、配置、源码快照及审计均保留。`MANIFEST.json` 对包内各成员（自身除外）记录 SHA-256。SVG 的生成器空白和旧科学源码快照保留原始字节，不为格式美化改写冻结结果。

## 范围与发布状态

作者原始九章协方差不在本包内；历史外部模型复现仍须依来源说明独立取得并核验。第八、九轮的新固定实验使用包内合成输入，不依赖该原始文件。

本轮没有向公开 GitHub 推送。此前自动审批审查拒绝了将这批研究内容与历史公开到 `qiyuanhuakai/deziroGBS/main`，要求明确批准具体内容和目的地；本包用于审阅与恢复，不代表已公开发布。研究与本地提交已完成，不受该公开发布状态影响。
