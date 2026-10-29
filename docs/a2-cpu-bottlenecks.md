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
- A1 的 pybind11 类方法触发了 cProfile 调用栈缺失：部分函数调用数丢失，甚至 self 时间大于累计时间。初次 cProfile 记录只保留作诊断，不用于占比或可靠调用数。改用独立边界计时，并检查调用次数和未剖析/剖析状态一致；累计时间仍相互嵌套，不能相加。速度改善另用未剖析基准证明。[pybind11 issue 5600](https://github.com/pybind/pybind11/issues/5600)
- 独立记录阶段 wall/process CPU 时间、缺页与上下文切换。预热为另一个实例，正式实例首次启动原生线程池包含在计时内。
- PMU 探测只检测权限/可用性，不把 `perf stat true` 或整个 Python 启动进程的计数当作阶段 IPC、缓存或主频证据。没有阶段硬件计数和有效频率测量时，不能声称“已证实主频受限”或“已证实内存受限”。
- Hyper-V 默认不向来宾开放 Intel PMU，容器还可能受权限和 `perf_event_paranoid` 限制。此次不调整宿主安全设置；若不可用则如实记录。[Microsoft Hyper-V PMU](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/manage/performance-monitoring-hardware)、[Linux perf permissions](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)

## 执行证据

### 迁移前瓶颈

`6badfd7` 的[云端边界计时](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40960718586)覆盖四个场景及 1/4 线程预算，所有调用完整性、状态和 RNG 比较均通过。[日志 JSON 记录](performance/a2-boundary-profile-6badfd7.json)。

| 场景 / 1 线程 | 未剖析阶段时间 | 重要调用数 | 原生批评分边界时间（剖析运行） |
|---|---:|---|---:|
| 人类 1,000 × 6 | 0.28786 s | 8,830 次选人，6,070 次有效交互，37,363 次 dominant 查询 | 0.00031 s |
| 人类 10,000 × 3 | 1.05602 s | 15,000 次选人，14,120 次有效交互，216,333 次 dominant 查询 | 0.00274 s |
| 人类 1,000 × 6，轮回者 | 0.30529 s | Guide 的 Adam 确实更新 6 次 | Guide 边界 0.01929 s |
| 永劫 200 × 2 | 0.02943 s | 152 次永劫交互 | 永劫交互边界 0.02531 s |

表中两类时间来自不同运行，不能相除当作精确 CPU 占比；剖析本身放大了小调用开销。10,000 场景的原生评分在 4 线程时缩短到 0.00170 s，但未剖析全阶段为 1.16135 s，没有整体收益。这支持优先移走交互循环、缓存读取和小调用边界，而非继续给批评分加线程。

云端 PMU 探测记录 `perf_event_paranoid=4`，cycles/instructions/cache-misses 无法访问，因此此次没有阶段 IPC、缓存命中率或有效频率的实测因果结论。CPU 为 runner 暴露的 AMD EPYC 9V74；静态 MHz 不是实际睿频对照实验。

### 脱离 pybind11 的边界

生产核心改为 C++17 共享库，公开有效的 C 头文件，不使用 Python C API 或 NumPy/pybind11 头文件。`ctypes` 只转换缓冲区和管理生命周期。无机阶段一次处理有序网格段；实体交互阶段一次处理本代全部交互，保留旧桶、淘汰顺序、缓存刷新、数组别名及两套 MT19937 状态。Python 仍保留模型后端、出生/演化的外层编排、事件显示与存档，尚不是整个应用的 C++ 重写。

中间提交 `534c984` 仅把小函数的接口换成 C ABI，人类循环还在 Python，云端同机对比出现约 1.7～2.8 倍耗时回退。这个版本不是最终提速成果。相同提交已移走完整无机循环，在 10×10、10,000 代场景分别取得 Linux 8.84、Windows 7.68、云端 macOS 11.46 倍速度，并且遗物和已检查 RNG 摘要相同。[中间版本对照](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40963299164)。这验证了“整段迁移”与“只更换绑定”的区别，后续人类阶段结果须使用完整迁移后的新证据。

实现和验收进度见[实施计划](superpowers/plans/2026-10-30-a2-profile-migration.md)。

### 完整交互迁移后的第二个瓶颈

`ef19ce1` 的[云端剖析](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40965514557)确认每个正常人类世代均成功进入 C++。10,000×3 的交互入口为 0.55155 s，其中资格检查 0.45725 s，C++ 执行为 0.00776 s，余下约 0.08655 s 为打包、写回等。1,000×6 对应为 0.08654 / 0.06939 / 0.00254 s。[已保存记录](performance/a2-encounter-bridge-ef19ce1.json)。这些是同一次剖析内的嵌套边界，可以定位入口成本，不能当作未剖析整阶段的加速倍率。

因此下一项优化是合并相同的类级方法检查，以及在连续状态缓冲区上批量验证有限值，保留实例覆盖、布局、所有权、共享数组和异常值的回退条件。不能因为 C++ 循环快就忽略 Python 入口仍需逐对象检查的成本。

日志回放保持内建日志消息的文本、顺序和事件时数值，但输出发生在本段状态提交后。自定义日志过滤器、handler 或 LogRecordFactory 若读取中间世界状态、消耗随机数或改变实体，不属于当前整段迁移的等价保证；这些插件需使用参考后端或单独适配。另有 Python 线程、替换 RNG、自定义实体/模型/处理方法、非标准数组等情况会保守回退到原有顺序循环。

`5d48c8c` 合并检查后，三平台功能回归全部通过：Linux/macOS 317 项，Windows 305 项加 12 个 POSIX 跳过。未剖析同机对照中，人类 1,000×6 提升 1.49～1.78 倍；10,000×3 的 Linux/macOS 为 1.14/1.03 倍，Windows 仍为 0.933 倍。这个残余回退促使继续合并选举阶段逐实体的缓存映射与主导路径读取，投票求和顺序保持不变。[此轮对照](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40966308801)。
