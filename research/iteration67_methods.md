# 第67轮方法与复核入口

冻结方案见[iteration67_cost_plan.md](iteration67_cost_plan.md)。所有正式源码在第一次运行前快照；初次验证的非二进制有理权重问题和新增恢复源码分别保留。

1. `iteration67_costs.py`生产20个理想核、140个有限律。c=1/4、α∈{1,4,16,4n²}、t∈{16,32,64,128,256,512,1024}固定。边权为αc²或α；公共链构造器统一分数化。
2. `verify_iteration67_costs.py`初次验证失败记录保留。`verify_iteration67_costs_recovery.py`以独立集合核、边组合枚举、反向奖励递推核验所有案例，输出`iteration67_costs_recovery_validation`。
3. `iteration67_select_recovery.py`及`verify_iteration67_selection_recovery.py`只使用已经核验的完整候选集合，认证10个误差门槛请求的20条路线。
4. `iteration67_samples_recovery.py`保留每次失败尝试和成功尝试的全部状态与随机字；每条路线7个公开输出。21项确定性控制强制发生选边拒绝尾部，其中1项另强制发生4/9接受判断的拒绝尾部。
5. `verify_iteration67_samples_recovery.py`用独立集合算法重放每一步并核对种子、停止条件、投影和总资源。`iteration67_independent_recovery.py`以分数单位元计算权重比，避免整数真除法。
6. `iteration67_analysis.py`画出同误差门槛的期望字数和转移数。`iteration67_tests.py`在前序发布更新后运行全库测试；`iteration67_seal_recovery.py`封存成功证书、原失败、预检ZIP及全部依赖。

一步期望字数为1加非懒惰时选边期望成本，再加各边需要有理Bernoulli时的成本。区间[0,b)的拒绝实现每次需要期望2^64/(2^64−2^64 mod b)个字，b=1时为0。前向累积与反向价值分别建立，无需用样本均值估计期望。最后除以单次成功质量得到独立重启成本；条件化完整有限律给出最终输出律。

12项探索预检完整归档为`results/iteration67_pilot/pilot.zip`，其原始文件名、源码快照和清单逐字节保留。预检选择最少转移数，正式计划选择最少随机字，两者的目标差异有意预先声明。

有理数结果保留在压缩JSON内。图只用小数展示，不作为严格证据。研究限于特殊无碰撞条件目标及最多140状态的精确枚举；程序局部执行不枚举状态，但目前误差证书仍依赖小规模枚举。

第66轮最终提交并入后，再次恢复并验证稀疏范围外的旧测试依赖。最终全库784运行、755通过、0失败、0错误；29个用例和7个类的既有跳过保留。首次封存因误将本轮新增验证器按旧提交依赖读取而失败，记录保留。最终使用iteration67_tests_recovery.py重跑完整测试，710份源码及快照核对一致，新增单元测试0项；使用iteration67_seal_recovery2.py显式处理本轮源码依赖，收录原失败与全部成功证据。原始全库测试同样通过，没有覆盖其记录。
