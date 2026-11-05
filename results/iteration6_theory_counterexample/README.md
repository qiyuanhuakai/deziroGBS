# 一个真实 Gaussian 的局部非单调性反例

这是一项**事后探索的确定性反例**，只说明相邻 pair 分块不保证比 singleton product 更接近某一个完整条件概率。它不属于第六轮冻结的 18 个正式 Haar 输入，不声称完整 TV 排序或统计确认，没有新采样与性能计时。

- `record/counterexample.npz`：保存 U、V、squeezing、transmissivity，以及所有 128 个阈值位串的完整目标 P。
- `record/case.json`：保存随机数生成顺序、种子、测量前缀、R/F/分块、两支事件质量、相关修正 C、结果、数值诊断与来源/数组散列。
- `reproduce.py`：独立由完整 P 求和的计算器，不导入 proposed `positive_completion` 模块。
- `implementation_check.json`：发现与独立计算完成后，对冻结新模块的 Gaussian direct / 同一 Gaussian 选集下的 table oracle 所作四次接口核对；不是用于选择反例的计算器。

在仓库根目录复现到一个尚不存在的目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python results/iteration6_theory_counterexample/reproduce.py \
  --output results/new_iteration6_theory_counterexample
```

关键前缀为 `(1,1,1,1,1,0)`，当前模式 6；recent k=2 保留 R=(3,4)，遗漏 F=(0,1,2)。目标条件概率为 0.14558312733059908，product 为 0.1451529488449504，pair `((0,1),(2,))` 为 0.1437242001201385。两个绝对误差分别为 0.0004301784856486912 与 0.0018589272104605759。

完整目标表没有负概率截断或归一化改动，但仍来自浮点行列式与包含排除；“完整”不代表严格区间证书。理论解释、Gaussian 支持结论及复杂度边界见 `research/iteration6_theory_audit.md`。
