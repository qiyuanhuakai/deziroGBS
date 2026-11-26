# 精选结果

## 第 98 轮

- [16、128、1,152 端口闭环的速度、完整分布误差界及准备时间](selected/98/ring.json)
- [同一 16 端口输入的 Frugal 和 photonic-engines 实测](selected/98/baselines.json) · [干涉控制、六光子样本及九章奇偶对照](https://github.com/qiyuanhuakai/deziroGBS/blob/a0f7a455fe024fe271b5cb1bfe0ca430cff90ca9/results/iteration98_release_v1/validation.json)

## 第 97 轮

- [连通单元的完整光子数与点击误差、七批计时](selected/97/forest.json) · [与 Frugal 和 photonic-engines 的同输入比较](https://github.com/qiyuanhuakai/deziroGBS/blob/d99c25e2cdb0f4f0c24dcc53f7a6efb7963fde9f/results/iteration97_bridge_baselines_v1/summary.json)
- [九章约化 16 位的首批计时和位移场模型的全分布界](selected/97/validation.json) · [真实 1,152 位硬件对照](https://github.com/qiyuanhuakai/deziroGBS/blob/d99c25e2cdb0f4f0c24dcc53f7a6efb7963fde9f/results/iteration97_field_v1/summary.json)
- [推导、适用范围和运行示例](https://github.com/qiyuanhuakai/deziroGBS/blob/d99c25e2cdb0f4f0c24dcc53f7a6efb7963fde9f/research/iteration97_checkpoint_v1.md)

## 第 96 轮

- [与 Frugal 同输入的完整 16 位点击误差](selected/96/click-validation.json) · [五次重复计时](https://github.com/qiyuanhuakai/deziroGBS/blob/393d3592b8e8013be76844cca88605f428666d39/results/iteration96_frugal16_v2/summary.json)
- [与 photonic-engines 同输入的 16 模式光子数计时与精度](selected/96/pnr-comparison.json)
- [九章 1,152 位的总点击数误差、置信区间与批量耗时](https://github.com/qiyuanhuakai/deziroGBS/blob/393d3592b8e8013be76844cca88605f428666d39/results/iteration96_calibrated_v1/summary.json) · [实验条件与复现方法](https://github.com/qiyuanhuakai/deziroGBS/blob/393d3592b8e8013be76844cca88605f428666d39/research/iteration96_checkpoint_v1.md)

## 第 95 轮

| 要查的问题 | 汇总文件 |
|---|---|
| 1152 端口的批量速度、单条延迟和总数误差是多少？ | [两条路线的计时与精度](selected/95/sampling.json) |
| 光子数联合分布和跨环点击的误差有多大？ | [独立光学核验](selected/95/optical-validation.json) |
| 和 Frugal 在同一组 144 个探测位上相比有多快？ | [同输入计时与首八位分布](selected/95/frugal144.json) |
| 两种算法生成的总点击数与九章实测相差多少？ | [144 位硬件对照](selected/95/frugal144-validation.json) |

这四份 JSON 逐字节保留归档版本。[实验条件与历次结果](../docs/reports/95.md) · [取回原始样本和清单](../docs/archive.md)

## 第 84、94 轮

- [第 84 轮重新采样的总光子数误差](iteration84_photon_cdf_v1/summary.json)
- [第 84 轮随机流复核](iteration84_rng_verification_v1/validation.json)
- [第 94 轮独立统计核验](iteration94_verification_v1/validation.json)
- [41 个配置的原始三次计时及中位数](iteration94_timing_v1/summary.json)
- [1152 端口双谱 PNR 的 64 次尝试](iteration94_actual_quality_v1/summary.json)
- [同一模型的直接点击失败统计](iteration94_branch_actual_quality_v1/summary.json)
- [Frugal 的实际工作种子复核](iteration94_rng_audit_v1/validation.json)
- [全库测试结果](iteration94_tests_v1/validation.json)
- [完整点击树与高精度投影检查](iteration94_branch_audit_v1/validation.json)
- [速度与点击误差对照图](iteration94_figures_v1/click_frontier.png)

[如何取回原始数据](../docs/archive.md)
