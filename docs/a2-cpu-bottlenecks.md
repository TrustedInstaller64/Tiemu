# A2：单世界 CPU 瓶颈与迁移记录

## 测量问题

本轮以 `4a2311dfff44e5b92829dfdb84f735f37c171701` 为 A1 基线，先剖析剩余开销，再迁移有证据支持的边界。C 暂缓且保持独立。所有项目运行、编译、测试和计时均在 GitHub 托管 runner；本地只进行静态检查、源文件编辑和小型证据文件操作。

“当前主要使用一个核心”不能推出“只有提高 CPU 主频才能提高速度”。对固定工作量，可以用近似关系理解：

`CPU 时间 ≈ 执行指令数 × 平均每指令周期数（CPI）÷ 有效频率`

C++ 移植和取消重复计算可以减少指令数；减少对象跳转、分配和缓存失效可以降低等待；提高有效频率是另一条路径。IPC、缓存和实际睿频不同的 CPU 不能只比较标称 GHz。这里不根据云端数据给 6138、8259CL、8280 或 M1 排采购名次。[Intel CPU metrics](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-2/cpu-metrics-reference.html)

`process CPU time / wall time` 接近 1 表示平均约一个核心的 CPU 时间。它不能区分 Python、原生单线程算术和内存停顿，也不能排除短暂多核工作与等待交替。CPU 占用率不能直接代表执行单元饱和程度。[Intel PCM](https://www.intel.com/content/www/us/en/developer/articles/tool/performance-counter-monitor.html)

当前批评分已释放 GIL，辅助线程只读写原生数据；Python 状态与对象访问仍受 GIL 约束。把循环语法搬到 C++、但继续频繁调用 Python 对象，不会自动得到有效多核。独立行可并行；下一交互读取上一交互状态、每步训练读取上一步权重时，维持 A 的结果必须保留这个顺序。[Python 3.12 GIL](https://docs.python.org/3.12/c-api/init.html#thread-state-and-the-global-interpreter-lock)

## 采证边界

- 四个有界场景：人类 1,000×6、人类 10,000×3、人类 1,000×6 且含轮回者、永劫 200×2。它们用于诊断，不是完整世界运行。Guide 是否实际更新由 optimizer step 明确记录，不因启用轮回者便假定训练发生。
- 每个场景对比 1/4 线程预算、未剖析/剖析运行的已检查状态摘要；原有固定 seed 和随机数顺序不变。
- cProfile 只用于定位与调用次数。累计时间相互嵌套，不能直接相加；剖析会放大小调用开销，速度改善必须另用未剖析基准证明。它不提供工作线程内部归因。
- 独立记录阶段 wall/process CPU 时间、缺页与上下文切换。预热为另一个实例，正式实例首次启动原生线程池包含在计时内。
- PMU 探测只检测权限/可用性，不把 `perf stat true` 或整个 Python 启动进程的计数当作阶段 IPC、缓存或主频证据。没有阶段硬件计数和有效频率测量时，不能声称“已证实主频受限”或“已证实内存受限”。
- Hyper-V 默认不向来宾开放 Intel PMU，容器还可能受权限和 `perf_event_paranoid` 限制。此次不调整宿主安全设置；若不可用则如实记录。[Microsoft Hyper-V PMU](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/manage/performance-monitoring-hardware)、[Linux perf permissions](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)

## 执行证据

云端任务结果与实际迁移在验证后补充。当前仅有静态假说，见[实施计划](superpowers/plans/2026-10-30-a2-profile-migration.md)。
