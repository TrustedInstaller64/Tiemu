# Tiemu v1.0.0

2026-11-10 · 上游基线 YezQiu/Tiemu `691c6e0`

这个版本对应严格等价的 A 线：状态、随机流和每次 optimizer 更新都与参考实现一致。允许轨迹离散的 C 线实验不包含在内。

## 中文

### 这个版本做了什么

A1–A5 的主线工作是把模拟里实测较慢的路径逐步移进独立 C++17 库。A1 先做数值核、平台支持和 Docker；A2 去掉 pybind11，改成 C ABI + ctypes；A3 加入常驻 World；A4 加入驻留评分准备、黄金裔筛选和永劫候选索引；A5 比较了两个候选，最后保持 A4 的默认开关。

Python 侧仍然负责编排、四个 PyTorch 网络的在线训练、生命周期决策、控制台和快照。C++ 侧负责数值核、有序候选索引、直方图、评分线程池、无机网格段、整代有序交互、常驻世界状态和出生批处理。

### 性能

成对固定阶段样本，不是整个程序或长期吞吐。倍率不能跨版本、跨场景相乘。

| 场景 | 对照 | 结果 |
|---|---|---|
| 人类 1,000×6 | A2 vs A1 | 2.10–2.30× |
| 人类 10,000×3 | A2 vs A1 | 1.32–1.47× |
| 无机 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| 永劫 200×2（真实训练） | A2 vs A1 | 0.944–0.958×（更慢） |
| 人类 10,000×3 | A4 vs A3 | Linux 1.068× / Windows 1.140× / 托管 macOS 1.041× |
| 永劫 1,000×12（full cycle） | A4 vs A3 | 1.438× / 1.610× / 1.506× |

有些场景更慢：A2 永劫 200×2 是 0.944–0.958×；A3 macOS 永劫 200×2 是 0.934×；A4 macOS 永劫 200×2/200×12 是 0.908/0.902×，macOS 无机是 0.751×。

1,000×12 的永劫样本每个 optimizer 只有 44 次更新（原循环在换宿主后丢了轮回者引用），不能当作持续训练提速。四线程没有稳定收益，默认还是 1。

### 升级时注意

- `world.json` 仍然不能恢复。它缺优化器、RNG 和阶段内部状态，`load` / `--load-save` 会拒绝，也不会覆盖现有输出。
- 改过 C++ 之后要重新 `pip install ".[test]"`。
- 依赖固定 NumPy 2.2.2、PyTorch 2.12.0；Python 3.11–3.13。
- 等价性测试需要完整 Git 历史，浅克隆先 `git fetch --unshallow`。
- 不同设备、精度、量化或 LLM 采样不保证逐位一致。

### 还没解决的问题

- 不是纯 C++：模型后端、部分生命周期决策、在线训练调度、控制台和快照还在 Python。
- 多核只用于独立评分，默认 1 线程；交互、抽样、投票和训练更新不并行。
- A5 的邻居预计算候选没有默认开启（实际无机阶段 Linux 0.998×、Windows 1.023×、macOS 0.986×）。
- Windows/CUDA、Hyper-V GPU 直通、ANE/Core ML 没有真机验证；没有 Core ML 后端。
- 没有多世界、批量实验或参数搜索调度器。

### 测试

196 个测试函数。A5 基线里 Linux/macOS 各 613 通过，Windows 601 通过加 12 个平台跳过；独立 C/CTest、单/四线程 CLI 和 Linux CPU Docker 通过。等价性用同一平台的冻结上游 `691c6e0` 回放，离散决策和 RNG 严格比较，世界值容差约 `1e-10`，float32 网络约 `1e-6`。

## English

v1.0.0 is the first formal release and corresponds to the strict-equivalence A line; the divergent C experiments are not part of it.

The release moves the measured heavy paths into a standalone C++17 library: numerical kernels, ordered candidate indexing, histograms, the scoring worker pool, inorganic grid segments, ordered encounters, the resident world state and batched births. Python still owns orchestration, the four PyTorch networks and their online training, lifecycle decisions, the console and snapshots. The boundary is a C ABI; pybind11 is gone since A2.

In paired fixed-phase samples, human 1,000×6 ran 2.1–2.3× faster than A1, human 10,000×3 1.32–1.47×, and inorganic 10×10×10,000 9.1–13.5×. Aeonic 200×2 with real training was slower at 0.94–0.96×. Against A3, A4 ran human 10,000×3 at 1.068× (Linux), 1.140× (Windows) and 1.041× (hosted macOS), and aeonic 1,000×12 at 1.438×, 1.610× and 1.506×.

Those aeonic samples only perform 44 optimizer updates per optimizer, so they are not sustained-training numbers. Some macOS cases regress (0.908/0.902× for aeonic 200×2/×12, 0.751× for inorganic), and four scoring threads show no stable whole-phase gain.

Upgrade notes: `world.json` is still not resumable and `load` / `--load-save` will refuse it; reinstall after changing C++; dependencies are pinned to NumPy 2.2.2 and PyTorch 2.12.0; equivalence tests need a full clone.

Known issues: Python still handles model backends, part of lifecycle decisions, online-training scheduling, the console and snapshots; multicore is limited to independent scoring and defaults to one thread; the A5 neighbor-precompute candidate is not enabled by default; Windows/CUDA, Hyper-V GPU passthrough and ANE/Core ML are unverified; there is no multi-world or batch-experiment scheduler.

Tests: 196 test functions. The A5 baseline suite passed 613 tests on Linux/macOS and 601 with 12 skips on Windows. Frozen upstream `691c6e0` is replayed on the same platform with exact discrete/RNG comparisons and tolerances of about `1e-10` for world values and `1e-6` for float32 networks.

MIT licensed. Original project and contributions: YezQiu, HeyCrab3, AvalonRuFae, MoonofBridge24, Akanyi. Logging uses [pyclog](https://github.com/Akanyi/pyclog).
