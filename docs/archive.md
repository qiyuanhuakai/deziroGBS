# 取回原始数据

第 1–94 轮的完整工程固定在标签 `archive/through-iteration-94`，提交为 `0f87aa4b1fbb8baae4e66a456e1505bfd5564d05`。该快照保留全部 125,269 个受跟踪的结果文件，以及当时的报告、输入、失败样本、程序和文件清单。[浏览完整快照](https://github.com/qiyuanhuakai/deziroGBS/tree/0f87aa4b1fbb8baae4e66a456e1505bfd5564d05) · [整理前的项目首页](https://github.com/qiyuanhuakai/deziroGBS/blob/0f87aa4b1fbb8baae4e66a456e1505bfd5564d05/README.md)

**只看一轮**　先从[报告目录](reports/README.md)找到该轮，报告中的原始数据链接会打开固定快照。第 94 轮可直接查看[实际 PNR 数据](https://github.com/qiyuanhuakai/deziroGBS/tree/0f87aa4b1fbb8baae4e66a456e1505bfd5564d05/results/iteration94_actual_quality_v1)和[直接点击数据](https://github.com/qiyuanhuakai/deziroGBS/tree/0f87aa4b1fbb8baae4e66a456e1505bfd5564d05/results/iteration94_branch_actual_quality_v1)。

**只下载一个结果目录**　在新的目录运行以下命令。将最后一行的路径换成所需实验即可。

```sh
git clone --depth 1 --filter=blob:none --sparse --branch archive/through-iteration-94 https://github.com/qiyuanhuakai/deziroGBS.git deziroGBS-evidence
git -C deziroGBS-evidence sparse-checkout set results/iteration94_actual_quality_v1
```

**完整复现旧实验**　从现有仓库创建独立工作目录，能同时取回历史输入、原生库、实验脚本和原始验证清单。

```sh
git fetch origin tag archive/through-iteration-94
git worktree add --detach ../deziroGBS-round94 archive/through-iteration-94
git -C ../deziroGBS-round94 rev-parse HEAD
```

最后一行应输出本页首段的提交号。在该目录按对应报告的环境要求复现；需要重跑核验时请给输出目录起一个新名字。[第 94 轮核验程序](https://github.com/qiyuanhuakai/deziroGBS/blob/0f87aa4b1fbb8baae4e66a456e1505bfd5564d05/experiments/verify_iteration94_v1.py) · [早期归档与发布记录](publication-history.md)

**早期分卷**　旧版大数组分卷、发布时的核验清单及历史构建材料集中在同一快照的 [research_archive 目录](https://github.com/qiyuanhuakai/deziroGBS/tree/0f87aa4b1fbb8baae4e66a456e1505bfd5564d05/research_archive)。使用历史发布工具时，请在完整归档工作目录运行。

**后续轮次**　原始数据随轮次发布到固定归档。主分支只收录代表性汇总、关键图表及对应归档链接；新增报告放入阶段目录，首页更新最新一轮入口。

**第 95 轮阶段快照**　[按物理端口编号的点击记录](https://github.com/qiyuanhuakai/deziroGBS/tree/2cc6d6d48679969d44d4b1bcfb15343e4d183f6b/results/iteration95_ensemble_v2) · [七次重复计时](https://github.com/qiyuanhuakai/deziroGBS/tree/2cc6d6d48679969d44d4b1bcfb15343e4d183f6b/results/iteration95_ensemble_timing_v1) · [PNR 重排的配对记录](https://github.com/qiyuanhuakai/deziroGBS/tree/2cc6d6d48679969d44d4b1bcfb15343e4d183f6b/results/iteration95_order_extension_v2)。链接固定到提交 `2cc6d6d48679969d44d4b1bcfb15343e4d183f6b`。
