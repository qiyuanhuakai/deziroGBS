# deziroGBS / GBS Classical Lab

量子项目的经典 Gaussian Boson Sampling 研究。当前公开成果截至第八轮原生算法实验与第九轮理论/资源筛查，原始研究提交 `8675497ae2ff16f7f6fbaba4ad6eb0545c1612ff`。

## 完整源码、数据和历史

本仓库提供代码、报告和结果汇总的浏览副本；全部原始实验结果和原始 Git 历史保存在 `research_archive/iteration9/` 的完整 Git bundle 分片中。分片按字节切分，清单记录每片及合并后的 SHA-256，不依赖外部网盘。完整恢复后包含原始提交号、全部科学代码、测试、供应商源码与许可证、成功及失败实验记录。

```bash
git clone https://github.com/qiyuanhuakai/deziroGBS.git
cd deziroGBS
python restore_complete_history.py --destination ../deziroGBS-complete
cd ../deziroGBS-complete
git rev-parse HEAD
```

最后一条应得到上述原始研究提交。请在恢复后的完整仓库中执行构建、测试和原始实验复跑；当前浏览副本不展开全部原始结果目录。恢复脚本只读取本仓库内的分片、核验散列并创建指定的新仓库，不下载第三方内容。

原始原生构建资产位于 `native_builds/iteration8/`，对应的跨环境构建说明见 [LEGACY_NATIVE_REBUILD.md](LEGACY_NATIVE_REBUILD.md)。这些产物使用 `-march=native`，移植到其他机器需重新验证或重建；原始构建和验证记录保持原样。

## 当前结果

- 第八轮：268 项测试通过；72 个新完整分布、96 项交叉核对、12 个历史回归分布和 108 次完整原生计时全部完成。共享事件核改善小系统质量和数值可靠性，但相对较强 retained 基线慢 34.73–40.43 倍；没有速度突破。
- 第九轮：纯态核心加经典位移的已知分解、相干截断的 `TV <= sqrt(tail)` 预算及 42 行资源筛查完成。大模式稠密 Fock 展开被资源规模排除；新的压缩态采样器尚未实现。
- 没有宣称通用低复杂度 GBS、方法原创性或九章规模完整分布模拟突破。

先读 [第八轮报告](RESEARCH_REPORT_ITERATION8.md)、[第九轮报告](RESEARCH_REPORT_ITERATION9.md) 和 [项目完整说明](PROJECT_README.md)。

![第八轮质量与成本](results/iteration8_analysis/quality_cost.png)

![第九轮截断与表示规模](results/iteration9_visualization/core_budget.png)
