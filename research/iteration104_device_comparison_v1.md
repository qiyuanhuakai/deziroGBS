# 第 104 轮补充 · Frugal 同 GPU 对照与硬件记录顺序

在同一张 RTX 5080 Laptop 上，给定九章 3.0 高功率双谱输入的前八个探测位，光学校正路线每条用时 0.643 微秒，Frugal GPU FP32 K2、K4 分别为 3.788、4.112 微秒。每批 2,048 条，均以相同八个字节返回记录。前者生成并传回全机再选取探测位；Frugal 对指定宽度的两个光谱独立建表、采样，回传后按位合并。Frugal 的两谱采样阶段单独计时分别为 1.910 和 1.958 微秒/条。[原始计时与分阶段耗时](../results/iteration104_frugal_device_v1/summary.json)

| 方法 | 微秒/条 | 完整八位经验 TV | 总体 TV 统计区间 |
|---|---:|---:|---:|
| 光学校正 | 0.643 | 0.549% | 0–0.969% |
| 硬件校准 | 1.188 | 1.674% | 0.497–2.094% |
| Frugal GPU FP32 K2 | 3.788 | 9.386% | 8.715–9.806% |
| Frugal GPU FP32 K4 | 4.112 | 1.317% | 0–1.737% |
| Frugal GPU FP64 K2 | 4.218 | 9.289% | 8.589–9.709% |
| Frugal GPU FP64 K4 | 4.627 | 1.345% | 0.196–1.765% |

光学参考覆盖全部 256 种点击字。每个方法用 32,768 条试采记录选择检验事件，再用 262,144 条新记录计算经验 TV 和区间。区间下界来自留出事件，上界取经验 TV 的单侧有界差分界和合并低概率点击字所得界的较小值。各单侧检验使用 0.0001 的错误概率，两组八位面板的区间合计覆盖率至少 99%。K2 和光学校正的区间分离；K4 和光学校正的区间重叠。在另一组八位探测位 288–295 上，光学校正、硬件校准、Frugal FP32 K2、K4 的经验 TV 分别为 0.495%、1.483%、6.193%、1.049%。[两组面板全部数值](../results/iteration104_device_release_v1/validation.json)

## 144 个观测探测位

每批 64 条，选取前 144 位后返回相同格式。光学校正和硬件校准各用 **13.353 和 15.766 微秒/条**。Frugal GPU K2 的 FP32 为 **0.843 秒/条**、FP64 为 **1.235 秒/条**。三个完整重复的两谱采样阶段中位数分别为 0.843 和 1.234 秒/条。这里的观测宽度和批量与上表不同。144 位的精度证据来自已核验的投影；完整 144 位联合 TV 没有由这次计时估计。第 103 轮同输入上的单线程 Frugal CPU K2 为 0.138 秒/条。[GPU 测量与全部重复](../results/iteration104_frugal_device_v1/summary.json) · [CPU 原始对照](../results/iteration103_frugal144_v1/summary.json)

Frugal 使用固定版本的官方 CUDA 数值内核，沿用已存档的 Windows 主机构建兼容改动。两种数值精度都保留该版本的 CURAND 均匀随机数路径。使用原头文件和 MSVC 重建种子派生，再以独立的 MT19937-64 实现复核；本次 Frugal 计时、试采和验证的 704 个外部种子映射到 704 个不同的设备种子。[构建说明](iteration72_methods.md) · [种子复核与数值汇总](../results/iteration104_device_release_v1/validation.json)

## 硬件原始文件的连续区段

从原始二进制文件逐条还原 1,492,642 个总点击数，聚合直方图与固定存档逐项一致。按原文件顺序分成 64 段，每段 23,322 或 23,323 条。将相同的光学检验函数应用于每一段，精确名义光学模型与每段实测总点击数分布的 KS 下界均为正，范围 **2.630–3.953 个百分点**。这些是对已记录样本的数值下界，计算无需对采样时间或各段独立性作假设。文件前后两半的 KS 为 0.182 个百分点，总点击数的相邻记录相关系数为 0.000323。硬件总体的 99% 下界仍以独立同分布假设为条件。[逐段直方图与相关系数](../results/iteration104_record_order_v1/summary.json)

任意宽度受压缩超模采样的理论 TV 上界以各随机用途获得独立均匀整数为条件；实现采用按实验、光谱、用途及记录地址分离的 Philox 随机流。[第 104 轮主报告](iteration104_checkpoint_v1.md)

```powershell
python experiments/iteration104_frugal_device_v1.py new-gpu-comparison
python experiments/verify_iteration104_device_v1.py new-seed-check
python experiments/iteration104_record_order_v1.py new-order-check --raw "high power (1).bin"
```

GPU 对照需要原归档中的第 72/75 轮 Frugal 动态库及其 Windows CUDA/OpenMP 运行环境。
