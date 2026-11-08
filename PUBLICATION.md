# 完整本地 GBS 研究公开归档

公开目的地：[qiyuanhuakai/deziroGBS](https://github.com/qiyuanhuakai/deziroGBS)。

本次按照项目所有者的明确要求，公开本地全部独有 GBS 研究资料：研究代码、理论推导、查新与审计、正式和探索实验、原始结果、失败记录、源码快照、报告、构建记录及原始 Git 提交历史。多份相同工作副本和可重建的软件环境不重复存储；独有研究文件按内容核对并保留来源。

## 主要入口

- [项目总览与复现](README.md)
- [第二十一轮：原4个纯核心预算难例实际构造达标](RESEARCH_REPORT_ITERATION21.md)
- [第二十一轮发布验证](results/iteration21_validation_release/validation.json)
- [第二十轮：直接带状MPS与窄带失败诊断](RESEARCH_REPORT_ITERATION20.md)
- [第二十轮验证记录](results/iteration20_validation/validation.json)
- [第十九轮：最终态MPS存在性与构造障碍](RESEARCH_REPORT_ITERATION19.md)
- [第十八轮：高阶真空级数与强基线](RESEARCH_REPORT_ITERATION18.md)
- [第十九轮验证记录](results/iteration19_validation/validation.json)
- [第十轮完整研究报告](RESEARCH_REPORT_ITERATION10.md)
- [第十轮全部质量与计时数据表](results/iteration10_analysis/summary_tables.md)
- [第十轮完整项目门禁](results/iteration10_validation/validation.json)
- [第十轮独立证据审计](research/iteration10_independent_audit.md)
- `research_archive/local_materials/`：工作副本之外补回的草稿、图稿、诊断、参考缓存、运行日志及输入来源。
- `native_builds/`：实际研究构建产物和对应构建记录；跨平台请使用源代码显式重建。

## 2026-11-09：第二十一轮持续发布

完整科学提交`608161a2e3dfcb84ba4f8f35c7fe0588cc61efef`已通过远端引用、重新获取的文件树及GitHub分支API交叉核对，且最终文件树与分批上传前的完整暂存树一致。514个原始产物共1,905,383,549字节验证通过，见[第二十一轮发布回执](research_archive/publication/verification_iteration21_2026-11-09.json)。

按持续发布授权追加Gaussian基构造、原态投影误差推导、正式58项工作和1项试运行、独立递推参考及验证。原4个深层局域纯核心构造均达到0.1预算，Haar预算失败和超时完整保留。新增8项测试、6,629项产物检查通过；全库356运行、322通过，既有4失败、1错误及29案例/7类跳过不变。

4个大型NPZ按40MiB分卷公开，`publication_support/iteration21_large_artifacts.py restore`还原并核对原冻结哈希。分批上传验证后的产物到研究分支，完整上传后再普通快进更新主分支；不强制覆盖远端。首次验证流程的版本衔接问题见[审计记录](research/iteration21_audit_note.md)，最终采用`iteration21_validation_release`。

## 2026-11-09：第二十轮持续发布

公开提交`f4f2c43830af4e0fbe734d3a1a536120d4403f4d`已通过远端引用、重新获取的文件树及GitHub分支API交叉核对。292个原始冻结产物共615,706,200字节验证通过，见[第二十轮发布回执](research_archive/publication/verification_iteration20_2026-11-09.json)。后续Gaussian基构造的[第二十一轮计划](research/iteration21_plan.md)仅为待验证计划。

两个约188MB的NPZ超过GitHub普通文件上限，以10个40MiB以内分卷公开，附完整还原工具和分卷清单。`python publication_support/iteration20_large_artifacts.py restore`还原后与原冻结SHA-256完全一致；发布核验直接对已暂存分卷重组哈希。首次未发布的整文件提交仅在本地备份分支保留，公开历史从此前主分支正常快进，没有改写既有公开提交。

按既有持续发布授权，追加直接前沿构造代码、推导、60项工作（含4项预分配拒绝）、250条带宽扫描、56个MPS、20个小盒参考和1,792条观测样本。新增7项测试与4,324项产物核验通过；全库348运行、314通过，既有4失败、1错误及29案例/7类跳过不变。冻结字节由`experiments/iteration20_seal.py`核对后普通快进发布。原深层4个输入仍未解决，窄带候选的纯态失败下界与输出TV严格区分。

## 2026-11-08：第11至19轮持续发布

项目所有者明确要求持续探索并持续发布到GitHub。第11至19轮原始研究提交从`64f5fd7`至`cba52ff`按原历史发布至主分支，包含代码、报告、推导、配置、完整成功/失败结果、图表、源快照和验证记录。后续完整研究轮次继续核验后发布；不把发布成功等同科学问题已解决。

发布前核对远端主分支是本地研究头的祖先，使用普通快进推送，不强制覆盖远端。`publication_support/verify_rounds_11_19.py`按9份冻结清单核对已提交Git blob的SHA-256；记录见[本次发布前核验](research_archive/publication/preflight_2026-11-08.json)。

发布后通过GitHub分支API、远端Git引用与重新获取的主分支交叉核对，确认公开提交`b2199b2`及文件树与本地一致。4,478个冻结产物共608,891,780字节通过核验，见[发布完成记录](research_archive/publication/verification_2026-11-08.json)。

最新全库记录为341项运行、307通过、4失败、1错误、29个案例跳过，另有7个类级跳过。新轮针对性测试和产物核对通过；遗留Windows数值范围/容差、Unix resource导入和原生依赖问题仍如实公开。深层大模式MPS实际构造、可扩展高阶采样与完整有限精度保证仍未完成，详见各轮报告。

## 原始提交历史与独立恢复归档

原有研究提交通过合并保留，未改写作者、日期、提交消息或树对象。第 1—9 轮的原始研究头是 `8675497ae2ff16f7f6fbaba4ad6eb0545c1612ff`。其 4,169 个跟踪文件和全部此前提交另有一份独立恢复归档：

[公开归档提交 d5fa1679fbfbe2523af62750dd75bbdffddf6efc](https://github.com/qiyuanhuakai/deziroGBS/tree/d5fa1679fbfbe2523af62750dd75bbdffddf6efc)

该提交中的 `research_archive/iteration9/MANIFEST.json` 列出每一分卷和全部源文件。完整 bundle 大小为 68,173,945 字节，SHA-256 为 `8bb4650e2a17f15609c1d4ed220fb9e209a55078bf4b46c0b70f8818af08fb4d`。`restore_complete_history.py` 会核验分卷顺序、大小、SHA-256 和恢复后的 Git 提交。

第十轮及本次补漏资料通过增量 bundle 传输；相应清单、分卷和导入请求保留在公开传输提交的历史中。导入工具核验两份 bundle、最终源树、允许添加的发布文件和快进关系，再使用普通 Git 推送。若远端已发生变化，工具停止，不强制覆盖。主分支的科学代码和结果与增量清单的源提交逐路径一致；发布工具和复现构建资料有明确的额外路径。

## 结果与证据范围

第十轮完整项目门禁 298 项通过，51/51 个完整质量分布、76/76 项检查通过；66 次预定计时中 60 次完整、6 次数值失败。16 个具备完整质量和成本参考的比较没有得到质量与速度的联合优势。失败记录和不支持的质量推断均保留，不因公开发布而删减。

核心构造的 `construction_reference_tv` 与旧基线完整有限原生扩展律是不同证据范围。新核心的已执行批次成功，不等于全局原生失败质量为零，也不构成已超越九章或经典算法上限的证据。

## 第三方来源

上游 FrugalGBS、xsimd 源码继续保留原许可证与提交信息。作者分发的九章一代协方差输入保留原始字节、来源提交和 SHA-256，按第三方输入标记；相关实验只主张已实际验证的协方差模型范围。来源记录中的许可证观察保持原样，不推断额外授权。本项目自有新增代码尚未指定开放源码许可证。

历史草稿、不可达旧图和探索缓存用于保留研究过程，不替代各轮冻结源代码、配置和正式结果。
