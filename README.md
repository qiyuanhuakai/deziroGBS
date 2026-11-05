# deziroGBS：完整本地 GBS 研究公开归档

本公开传输提交包含第 1—10 轮研究、全部原始 Git 历史与本地补漏资料。项目原始研究目录由经核验的 Git bundle 导入 main；该传输提交永久保留全部恢复数据。

- [第十轮报告](RESEARCH_REPORT_ITERATION10.md)
- [完整项目说明](PROJECT_README.md)
- [公开范围、来源与历史说明](PUBLICATION.md)
- [第十轮质量与计时完整汇总](results/iteration10_analysis/summary_tables.md)
- 第 1—9 轮完整历史：`research_archive/iteration9/MANIFEST.json` 和 66 个分卷。
- 第十轮和 609 份独有本地补漏资料：`research_archive/iteration10/MANIFEST.json` 及其增量分卷；它以第九轮原始研究头为唯一前提。
- 导入请求：`.github/gbs-publication.json`；导入工具：`publication_support/import_history.py`。工具核验全部分卷、Git 原始提交及完整文件树，只作普通快进推送。

第十轮通过 298 项测试、51 个完整质量分布和 76 项交叉检查。66 个预定计时有 60 个完整成功、6 个基线数值失败。16 个质量和成本比较没有取得联合优势，所有失败及证据限制均公开。

原始历史不重写。完整 Git clone 后可通过本提交中的 manifests、bundle 和主分支合并的原始 source parent 独立核验公开内容。
