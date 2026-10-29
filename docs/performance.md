# 性能与等价性记录

## 2026-10-30：单世界 A 分支远端验证

本轮基线是已经带有 C++ 加速的 `1ce39e00b8808564a331861ffcc799d4237ac324`，候选是 A 分支 `22b1620a7a0e4676831e14b209ba9e1227f0aac0`。这些倍率表示相对上一版的增益，不能与下方 2026-10-28 不同场景的倍率相乘。所有编译、测试和性能测量均在 GitHub 托管 runner 进行；本轮没有在用户的 Mac 上运行这些任务或安装 Docker。

本轮增加的生产路径包括：有序 C++ 纯化/潮汐/裁剪/归一化融合；议会席位向量缓存与投票模板复用；固定全局分布后独立实体行的批评分和可关闭的 C++ 线程池；采样前向禁用梯度、共用状态 tensor 和移除多余计算图保留；保留日志参数副作用的 `--compute-only`。这是一轮实质迁移，尚未把整个 Python 引擎改写成 C++。

### 计时范围和原始证据

[成对性能任务](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40954826586)在同一 runner 上顺序启动隔离进程，分别构建并核对基线/候选原生扩展路径。每项测三次、轮换运行顺序、取中位数；PyTorch/BLAS 单线程，固定 seed=731 和 PYTHONHASHSEED=0。没有 LLM、日志、显示和导出。人类阶段无轮回者；永劫阶段执行真实、按原顺序进行的网络训练并检查 optimizer 已更新。这是回归测量，不是独立世界并行或批量实验功能。

正式样本从新建世界开始计时实际阶段，排除世界初始化、预热实例、最终状态抓取和进程启动；**正式实例首次达到评分阈值时启动线程池的成本包含在计时中**。因此这些短阶段数据不代表充分预热的长期线程池吞吐，也不代表完整世界或 LLM 延迟。三次短样本及共享 runner 负载不足以给出稳定的长程加速承诺。

已将 [Windows](performance/single-world-2026-10-30-windows.json)、[云端 macOS](performance/single-world-2026-10-30-macos.json) 的完整原始 JSON 保存到仓库，包含各次 wall/process CPU 时间、平台/架构/依赖版本、权重与 RNG 摘要及汇总，并核对了下载 ZIP 的 SHA-256。Linux 连接器返回的下载内容重试后仍不是有效 ZIP，未保存该内容；仓库改存由成功任务日志读取的 [Linux 汇总](performance/single-world-2026-10-30-ubuntu.json)，单次明细与机器元数据仍在 Actions artifact 中，受其保留期限限制。文件保留测量当时的原始 scope 字段；上述线程池启动边界是后续静态复核补充。

| 云端平台 | 场景：初始人口 × 代数 | 基线秒 | A / 1 线程秒 | A / 1 线程速度倍率 | A / 强制 4 线程速度倍率 |
|---|---|---:|---:|---:|---:|
| Linux | 人类 1,000 × 6 | 0.39773 | 0.28963 | 1.373 | 1.364 |
| Linux | 人类 10,000 × 3 | 1.41033 | 1.11083 | 1.270 | 1.334 |
| Linux | 永劫 200 × 2 | 0.02911 | 0.02824 | 1.031 | 1.032 |
| Windows | 人类 1,000 × 6 | 0.33163 | 0.24382 | 1.360 | 1.357 |
| Windows | 人类 10,000 × 3 | 1.19633 | 0.92546 | 1.293 | 1.249 |
| Windows | 永劫 200 × 2 | 0.03001 | 0.02958 | 1.015 | 1.002 |
| macOS | 人类 1,000 × 6 | 0.31489 | 0.25794 | 1.221 | 1.638 |
| macOS | 人类 10,000 × 3 | 1.18325 | 1.12335 | 1.053 | 0.996 |
| macOS | 永劫 200 × 2 | 0.02462 | 0.01622 | 1.518 | 1.353 |

四个配置为基线、A 单线程、A 四线程预算/默认 4096 行阈值，以及 A 四线程预算/强制阈值 1。线程预算包括调用线程；四线程最多创建三个辅助线程。只有批评分有多于一个有效实体且达到阈值才使用辅助线程。**永劫样本本身不调用该人类批评分池，表中“四线程”列不构成并行训练的证据。**

多核结论：当前收益主要来自融合和缓存。Linux/Windows 的全阶段进程 CPU 时间与 wall 时间之比仍约为 1，独立评分占完整阶段的比例较小，强制并行没有跨场景、跨平台的稳定额外收益。因此默认仍是 `--native-threads 1`；四线程是可选探索配置，不把线程数当作速度倍率，也不默认占满双路 Xeon 或 M1 的核心。

云端 macOS 波动显著：[上一轮](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40953921045) A 单线程人类 10,000 场景为 1.549 倍，本轮为 1.053 倍（[上一轮日志汇总存档](performance/single-world-2026-10-30-macos-earlier-summary.json)）。小人口默认 4096 阈值不会启用评分池，仍可见单/四线程预算配置间的耗时差异，也说明进程/宿主调度噪声不能归因于线程池。这里不报告这台 M1 的新倍率，不用云端短样本推断 Hyper-V、NUMA、Metal、CUDA 或 ANE 性能。

### 正确性及验收边界

- [远端 RED](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40952329166)：原有 131 项通过；新增契约得到 93 个预期失败和 18 个通过，失败原因限定为未实现接口、集成和梯度开销等预定项。
- [最终功能回归](https://github.com/TrustedInstaller64/Tiemu/actions/runs/40954826520)：Linux 245 通过；macOS 245 通过；Windows 233 通过、12 个 POSIX 项跳过；Linux CPU Docker 构建及烟测通过。原生 1 线程与强制 4 线程烟测均通过。
- 新增差分覆盖 1/2/4 线程逐行结果、空输入/非有限值/只读和非连续数组、非法形状、并发调用及关闭；保留非正纯化因子的外部数组别名副作用、旧路径 multiplier 读取时点和席位字典直接修改语义。
- 同平台冻结原版回放继续使用下方已有容差与严格离散/RNG 断言；新增真实训练差分比较动作、参数、完整 Adam 状态和 Python/NumPy/PyTorch RNG。外部 LLM 使用确定性桩，不能据此声称真实模型的跨后端输出相同。
- 性能样本的所有配置均得到完全相同的**已检查状态摘要**：实体序列化内容及顺序、蓝图、席位、网络权重、Adam 状态、三种 RNG。摘要不包含每个管理器的全部内部字段，不是任意长轨迹逐位等价证明。
- 独立静态审查发现 compute-only 会隐藏暂停后的 help/status 输出；现已临时恢复暂停命令的日志级别并在退出后恢复静默，真实命令解析测试及上述远端回归通过。

### 分支和下一步

A 位于 `codex/single-world-a`，以保持原规则和顺序为先。C 位于独立的 `codex/experimental-c`，当前只提交[长期实验与差异登记](https://github.com/TrustedInstaller64/Tiemu/blob/codex/experimental-c/docs/experimental-c.md)，尚未实施降低精度、批训练或 GPU 世界计算，也没有合并到 A/main。

继续 C++ 迁移应先远端剖析剩余 Python 状态准备、实体操作和交互调度，再逐段建立稳定实体编号、连续状态数组及 C++ 事件循环。可以并行的只读行计算、快照序列化等需先明确状态生命周期；依赖上一交互结果的世界更新和每步在线训练保持顺序。完整移除 Python、跨 NUMA 并行和异步快照均尚未完成，不能由本轮线程池存在推断其收益。

## 2026-10-28：历史 M1 基线

以下是本轮禁止本地高负荷运行之前留下的记录，未在 2026-10-30 本地重跑。

原版基线为 `691c6e0023a83513ad3cd386cfee2198317104d3`，使用 `git archive` 独立提取。测试运行在 Apple M1、Python 3.12.1、NumPy 2.2.2、PyTorch 2.12.0、CPU 单线程；LLM、日志、显示及键盘输入在计算基准中关闭。每组预热后测 3 次，以下取中位数。所有计时包含实际 Python/C++ 调用、数据转换和返回开销。

## 计算基准

| 场景 | 原版 | C++ 路径 | 加速比 |
|---|---:|---:|---:|
| 人类阶段：1,000 初始实体、2 代 | 3.0392 秒 | 0.09601 秒 | 31.65 倍 |
| 无机阶段：10×10 网格、1,000 代 | 0.15482 秒 | 0.02486 秒 | 6.23 倍 |

这是**阶段基准，不是完整世界总耗时或长期训练加速比**。人类场景使用 seed=731、中性遗物、无轮回者，计时包含真实 `_run_one_generation` 与 `_evolve_and_grow`，排除启动和早期阶段。两条路径最终人口均为 722；无机遗物均为 activity=3.2958931919023873、stability=2.8269341421750496。

原始记录：[原版人类](performance/original-human-1000.json)、[C++ 人类](performance/native-human-1000.json)、[原版无机](performance/original-inorganic.json)、[C++ 无机](performance/native-inorganic.json)。

最初的分组测量在人类场景中记录约 73.83% 的全局路径统计、19.15% 的选人耗时；小矩阵映射本身只占较小部分。它们都是含原生调用的整条路径耗时，不能声称“95% 都是解释器”。主要收益来自有序候选索引、作用域内增量路径统计，以及 C++ 数值核，而不只是把小矩阵拼大。[分组原始记录](performance/original-human-profile.json)。

无机循环保留原版浅拷贝导致的顺序原地更新；人类阶段保留每代 `min(2*N,5000)` 次交互、原候选顺序和旧桶中仍保留已淘汰实体的行为。未合并 optimizer step，未改变精度或批量，也未跳过任何演算来换取速度。

永劫阶段的有限原版人工初态 profile 表明，PyTorch forward/backward/Adam 已占显著成本；这些本来就调用原生算子。不能沿用人类阶段的占比，也不能把多次在线训练合并成一次批训练而仍宣称等价。

## Mac 小网络与 LLM

固定网络形状、float32、预热 20 步后 100 步×5 次，MPS 在计时边界显式同步，包含当前调用路径的 CPU/设备数据移动及 `.item()` 依赖：

| 网络 | CPU 单线程/步 | MPS/步 |
|---|---:|---:|
| Guide（49→128→256→128→12） | 1.128 ms | 4.936 ms |
| Action + Value（单样本、24 维输入） | 0.975 ms | 9.339 ms |

因此当前默认 CPU 小网络、1 个 PyTorch 线程，保留 MPS/CUDA 显式选择和线程数配置。不宣称此配置在所有机器上最优。[完整结果](performance/apple-m1-networks.json)。

另在测试结束后顺序补测 CPU 的 1/2/4/8 线程，每组 100 步×5 次：Guide 分别为 0.597/0.635/0.633/0.741 ms，Action+Value 为 0.673/0.663/0.694/0.637 ms。多线程没有对两类网络都获益，保留默认 1 线程，用户可用 `--torch-threads` 在实际主机复测。这是另一批测量，不能与上面的 CPU/MPS 表拼接计算倍率。[线程测试记录](performance/apple-m1-cpu-threads.json)。

本地 Qwen3-0.6B-MLX 已完成真实 Metal 加载和生成：加载 4.885 秒，最多 16 tokens 的请求生成“你好！”用时 1.105 秒；模型不随仓库提交，没有隐式下载。这是功能验收，**不是 LLM 吞吐量基准**。[验收记录](performance/apple-m1-mlx.json)。

MLX 不使用 ANE。当前项目没有 Core ML/ANE 后端，也未测功耗。固定模型将来可转换为 Core ML，再验证 CPU+ANE 分配与能效；在线训练和世界规则不能靠一个 ANE 开关迁移。[Apple MLX](https://developer.apple.com/videos/play/wwdc2025/315/)、[Core ML 计算单元](https://developer.apple.com/documentation/coreml/mlcomputeunits)。

## 等价性证据

- 原生单元差分：路径映射、概念评分、纯化、范数上限、投票、顺序网格、4 种交互、直方图、候选池边界；覆盖零值、非有限值、形状错误和随机删除。
- 冻结原版回放：人类/引导各 40 初始实体、3 代；无机 17 代；永劫 100/200 实体各 2 代。各 CI 平台现场重跑固定 SHA 的原版作为对照，Python 和 C++ 均与之比较；仓库 JSON 保留 M1 上的测量样例，不作为跨架构逐位标准。
- 世界浮点容差 `1e-10`、网络 float32 输出约 `1e-6`；离散决策、人口顺序（阶段回放）、选举及 Python/NumPy/PyTorch RNG 状态使用严格比较。
- 永劫回放实际训练 Action/Value，每个网络分别执行 12/15 个 optimizer step，并验证参数确实改变，覆盖重建和轮回者淘汰分支。
- 截至本地最终验证，完整测试 `131 passed`；另有 POSIX 真 PTY、无终端 CLI、SIGTERM 输出保存、安装 wheel 后离开源码目录运行以及 CPU 权重读取验证。

首轮 CI 暴露了原版已有的并列投票排序差异：智识/记忆票份额均为 0.08089141932730622，未指定稳定性的 `np.argsort` 在 M1/x86 上选择了不同的剩余席位。保持原规则意味着不能把 Mac 的席位结果硬套到 Linux/Windows；因此改为同平台运行冻结原版，再严格比较席位及 RNG，不放宽离散断言。Windows 的 float32 探针同样在本平台生成。

短轨迹和有限参数不能证明任意长时间随机演化逐位一致。不同 CPU 数值库、PyTorch 版本、GPU 随机数、浮点误差、模型量化和 LLM 采样均可能改变后续分支。此版保持原规则，不承诺 CPU/CUDA/MPS/MLX 的所有推理输出完全相同。

## 复现

```sh
python -m pip install ".[test]"
python -m pytest -q
python benchmarks/benchmark_simulation.py --backend cpp --population 1000 --generations 2 --repeats 3
python benchmarks/benchmark_simulation.py --backend python --phase inorganic --generations 1000
python benchmarks/benchmark_networks.py --devices cpu mps --threads 1 --steps 100 --repeats 5 --output network-results.json
```

对**原版**基准须先将固定提交提取到单独目录，再给 `benchmark_simulation.py` 传 `--source-root /原版目录 --backend python`。当前 `--backend python` 是保留算法的参考路径，仍带新的控制和设备层，不能冒充原版源码计时。更换 `--seed`、规模、版本、模型或日志配置时应重新测量。

## 平台验收范围

本机验证了 macOS C++/CPU、POSIX 控制台、真实 MLX Metal、真实 MPS 前反向和更新。Windows 控制台逻辑使用模拟 API 测试；Windows/MSVC、Linux/GCC 及 CPU Docker 有 GitHub Actions 工作流。用户明确要求不在这台 Mac 安装 Docker，本次未安装；容器验收在 GitHub Linux runner 进行，首轮已通过镜像构建和非 root CPU 烟测。Windows/CUDA、Hyper-V 宿主、GPU 直通没有真机验收。CI 的具体结果应单独查阅，不能把配置文件存在视为通过。

数值依赖固定为 NumPy 2.2.2、PyTorch 2.12.0，Docker 使用对应 CPU 构建；不同主机/数值库的速度与舍入仍需实测，不能把 M1 数字当作 Hyper-V Docker 的数据。JSON 仅是检查快照，`--load-save` 不支持恢复，详见[运行说明](../readme/readme_zh-cn.md)。

GGUF 后端检查与加载成功不代替 Windows CUDA 真机验收：公共绑定不报告实际已卸载层数，多 GPU 后端/RPC 的自定义构建需要检查实际设备、模型缓冲与短生成日志。当前默认 CPU Docker 不涉及这些 GPU 分支。
