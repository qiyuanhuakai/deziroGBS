# 第84轮方法与可复核范围

随机流地址、标准向量和十事件置信族见[专门说明](iteration84_rng_method.md)。

## 总光子数参考与误差

对零均值Gaussian态，X=(Gamma-I)/2的实特征值为lambda_i。总数PGF为
G(z)=prod_i(1+(1-z)lambda_i)^(-1/2)。两个独立谱的PGF相乘，即合并特征值；
被动损耗约化已包含输入损耗，不能再次添加损耗。

令c_i=lambda_i/(1+lambda_i)，p_0=prod_i(1+lambda_i)^(-1/2)，
a_j=sum_i c_i^j/2，则n p_n=sum_{j=1}^n a_j p_{n-j}。
实现用mpmath90位递推到1024；负c对应压缩方向，不能裁为零。
尾界使用正实s>1的Chernoff界P(N>1024)<=G(s)/s^1025，处于PGF收敛域。
独立验证使用4096个单位根上的复数PGF作Fourier反演，并核对迹形式的均值/方差。
输入特征值来自浮点协方差，这些检查不是区间算术证明。

每方法Dhat=sup_n|Femp(n)-Fref(n)|，epsilon=sqrt(log(2/delta)/(2N))。
在该方法已预分配的DKW一致事件上，真实CDF距离位于
[max(0,Dhat-epsilon),min(1,Dhat+epsilon)]，另计可忽略的参考尾界。
同一CDF事件可同时支持所有阈值和单调有界函数，未重复使用失败概率预算。
TV(full PNR)>=Dcount，但反向上界不成立。core0的5.5753%上界只能用于计数CDF。

## 整数点积累计

Weight83由两个binary128分量组成。每分量113位有效尾数，与x87 long double的
64位有效尾数相乘至多177位。新核分别累计正负乘积，在相对指数跨度不大于310时，
最多2*256项需要的位数不超过310+177+9=496，装入8个64位limb。
整数减法后按limb转回原权重，继续原来的除法和归一化。
原浮点输入的整数乘加阶段精确，不意味着最终转换/条件概率具有区间认证。
更宽指数跨度及转换异常保护回退旧点积，不能静默丢失小项。

本实现限定GCC支持的16字节long double、64位尾数平台，链接quadmath。
`tc84e_sample`支持外部统一随机数；统计8列，前6列兼容83，后2列记录整数点积/回退次数。
`sum84_probe`供标量对照；243案例的独立验证用Python Fraction重建输入乘积。
优化是已知精确累计思想在现有权重表示中的应用，不宣称发明了通用精确GBS算法。

## 复验入口与冻结数据

各目录plan包含编译命令、源码哈希、模型输入和运行参数，manifest覆盖产物。
重采入口`experiments/iteration84_pnr_regenerated_v1.py`，见证入口
`iteration84_pnr_witness_v1.py`，总数参考入口`iteration84_photon_cdf_v1.py`。
正式点积比较入口`iteration84_exactsum_timing_v1.py`；它原生生成显式分域均匀数，
旧新共享随机数用于配对重放，不计作独立新质量数据。

复核现有归档可运行：

```text
python -X utf8 experiments/verify_iteration84_rng_correction_v1.py --output <新验证目录>
python -X utf8 experiments/verify_iteration84_final_v1.py --output <新验证目录>
```

程序拒绝覆盖既有结果；科学源码执行后冻结，任何修订需另建版本。完整测试快照覆盖
当时所有原生/src/tests源码；之后新增CDF分析与最终验证脚本由独立参考检查覆盖，
没有因文档更新重复运行995项测试。
