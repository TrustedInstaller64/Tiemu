# Tiemu v1.0.0 Release Notes

发布日期 / Release date: 2026-11-10
标签 / Tag: `v1.0.0`
上游基线 / Upstream baseline: `YezQiu/Tiemu` `691c6e0`

---

## 中文

### 概览

v1.0.0 是 Tiemu 的首个正式发布。它在保留上游世界规则的前提下，为演化模拟加入了独立的 C++17 数值核心、常驻世界状态、跨平台打包和差分验证。Python 仍负责编排、PyTorch 在线训练、生命周期决策、控制台和快照；C++ 核心通过 C ABI 与 `ctypes` 接入，不使用 pybind11。

### 亮点

- **独立 C++17 核心**：可脱离 Python 用 CMake 构建，提供纯 C ABI 供其他语言调用。
- **常驻 World**：连续数值列、稳定实体 ID、人口顺序、成员对账与出生数值批处理。
- **有序等价**：保留交互顺序、候选池顺序、旧缓存与数组别名语义、两套 MT19937 状态和每次 optimizer 更新顺序。
- **保守启用策略**：驻留评分默认启用；生命周期原生筛选和永劫候选索引按实测成本限定启用。
- **跨平台**：Linux、Windows、macOS CI；独立 C/CTest；Linux CPU Docker；单线程与四线程 CLI 烟测。
- **测试规模**：196 个测试函数；A5 基线默认回归在 Linux/macOS 通过 613 项，Windows 通过 601 项并跳过 12 项。
- **文档与证据**：A2–A5 设计/性能记录、迁移日志和 52 份性能证据 JSON。

### 架构

- Python 3.11–3.13：CLI、配置、阶段编排、模型定义与在线训练、生命周期决策、控制台、快照、LLM 后端。
- C++17：数值核、候选索引、直方图、评分线程池、无机网格段、有序交互、常驻世界状态、出生批处理。
- 桥接：C ABI 共享库 `_tiemu_core` + `_tiemu_ffi` 薄层适配器；无 Python C API、NumPy 头文件或 pybind11。

### 性能

以下均为固定阶段短样本，不是完整程序或长期吞吐承诺；不同版本、不同场景的倍率不能相乘。

| 场景 | 对照 | 结果 |
|---|---|---|
| 人类 1,000×6 | A2 vs A1 | 2.10–2.30× |
| 人类 10,000×3 | A2 vs A1 | 1.32–1.47× |
| 无机 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| 永劫 200×2（真实训练） | A2 vs A1 | 0.944–0.958×（变慢） |
| 人类 10,000×3 | A4 vs A3 | Linux 1.068× / Windows 1.140× / 托管 macOS 1.041× |
| 永劫 1,000×12（full cycle） | A4 vs A3 | Linux 1.438× / Windows 1.610× / 托管 macOS 1.506× |

完整表格、原始证据与平台限制见 [docs/performance.md](docs/performance.md)。

### 新增与变更

- 版本号由 `0.2.0` 提升为 `1.0.0`。
- CLI：`--backend {auto,python,cpp}`、`--device {auto,cpu,cuda,mps}`、`--torch-threads`、`--native-threads`、`--native-min-rows`、`--compute-only`、`--fast-forward`、`--no-interactive`、`--llm-backend`、`--llm-device`、`--gguf-model`、`--mlx-model`、`--seed`、`--output-dir`、`--smoke-test`。
- 默认 `--backend auto`：优先原生核心，缺失时警告并回退；`cpp` 明确报错；`python` 保留参考路径。
- 默认 `--torch-threads 1`、`--native-threads 1`、`--native-min-rows 4096`。
- `pip install ".[test]"` 才构建原生核心；仅安装 `requirements.txt` 不会。
- 新适配器 `_tiemu_ffi` 加载 `_tiemu_core`；旧 `_tiemu_native` 扩展不会被选中。
- 输出使用原子替换；默认目录为 `runs/`。

### 兼容性与升级注意

- **快照不可恢复**：`world.json` 仅是检查快照，缺少优化器、RNG 与阶段内部状态；`load` 和 `--load-save` 会拒绝恢复，且不会覆盖现有输出。这是相对上游的有意行为变更。
- 修改 C++ 源码后必须重新执行 `pip install ".[test]"`；旧 `.so/.pyd` 不会自动更新。
- 依赖固定为 NumPy `2.2.2`、PyTorch `2.12.0`，Python 3.11–3.13。
- 不同设备、精度、模型量化或 LLM 采样不保证逐位相同的长程轨迹；短样本回放不能证明任意长期等价。
- 完整等价性测试需要完整 Git 历史；浅克隆请先 `git fetch --unshallow`。

### 已知限制

- 这不是纯 C++ 重写：Python 仍保留模型后端、部分生命周期决策、在线训练调度、控制台和快照。
- 多核只用于独立评分；默认单线程，四线程没有稳定的整阶段额外收益，不并行交互、抽样、投票求和或训练更新。
- 部分阶段/平台变慢：A2 永劫 `200×2` 0.944–0.958×；A3 macOS 永劫 `200×2` 0.934×；A4 macOS 永劫 `200×2`/`200×12` 0.908/0.902×，macOS 无机 0.751×。
- A5 邻居预计算候选实际无机阶段为 Linux 0.998×、Windows 1.023×、macOS 0.986×，未默认启用。
- A4/A5 的 `1,000×12` full-cycle 样本每个 optimizer 仅 44 次更新，不是持续训练吞吐；原轮回者引用缺口被保留。
- Windows/CUDA、Hyper-V GPU 直通、物理 M1 后续复测和 ANE/Core ML 未验证；当前没有 Core ML 后端。
- 没有多世界、批量实验或参数搜索调度器。

### 测试与验证

- 196 个测试函数；A5 基线默认回归 Linux/macOS 各 613 通过，Windows 601 通过 + 12 项跳过。
- 独立 C/CTest、单/四线程 CLI、Linux CPU Docker 通过。
- 同平台冻结上游 `691c6e0` 回放；严格比较离散决策、RNG、权重与完整 Adam 状态；世界浮点容差约 `1e-10`，float32 网络约 `1e-6`。
- 发布验证：`main` 与 `v1.0.0` 指向同一提交；Release body 与本文件一致。

### 文档与许可

- [README](README.md) · [中文说明](readme/readme_zh-cn.md) · [性能与等价性](docs/performance.md) · [设计文档与迁移日志](docs/design/) · [世界参数指南](readme/Guide.md) · [日志说明](readme/clog_zh-cn.md)
- MIT License；原项目与贡献者：YezQiu、HeyCrab3、AvalonRuFae、MoonofBridge24、Akanyi。日志使用 [pyclog](https://github.com/Akanyi/pyclog)。

---

## English

### Overview

v1.0.0 is the first formal release of Tiemu. It keeps the upstream world rules and adds a standalone C++17 numerical core, resident world state, cross-platform packaging and differential validation. Python still owns orchestration, PyTorch online training, lifecycle decisions, console control and snapshots; the C++ core is reached through a C ABI and `ctypes`, without pybind11.

### Highlights

- **Standalone C++17 core**, buildable with CMake without Python and exposing a pure C ABI.
- **Resident World** owning continuous numeric columns, stable entity IDs, population order, membership reconciliation and batched birth arithmetic.
- **Ordered equivalence**: interaction order, candidate pools, old cache and array-alias semantics, both MT19937 streams and every optimizer update are preserved.
- **Conservative defaults**: resident scoring is enabled by default; native lifecycle selection and aeonic candidate indexing are gated by measured cost.
- **Cross-platform**: Linux, Windows and macOS CI; standalone C/CTest; Linux CPU Docker; single- and four-thread CLI smoke tests.
- **Test scale**: 196 test functions; the A5 baseline suite passes 613 tests on Linux/macOS and 601 with 12 skips on Windows.
- **Documentation and evidence**: A2–A5 design and performance records, a migration log and 52 performance-evidence JSON files.

### Architecture

- Python 3.11–3.13: CLI, configuration, phase orchestration, model definitions and online training, lifecycle decisions, console, snapshots and LLM backends.
- C++17: numerical kernels, candidate indexing, histograms, scoring worker pool, inorganic grid segments, ordered encounters, resident world state and batched births.
- Bridge: C ABI shared library `_tiemu_core` with the thin `_tiemu_ffi` adapter; no Python C API, NumPy headers or pybind11.

### Performance

All figures are fixed-phase short samples, not whole-program or long-term throughput promises. Ratios from different versions or workloads must not be multiplied.

| Workload | Comparison | Result |
|---|---|---|
| Human 1,000×6 | A2 vs A1 | 2.10–2.30× |
| Human 10,000×3 | A2 vs A1 | 1.32–1.47× |
| Inorganic 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| Aeonic 200×2 (real training) | A2 vs A1 | 0.944–0.958× (slower) |
| Human 10,000×3 | A4 vs A3 | 1.068× Linux / 1.140× Windows / 1.041× hosted macOS |
| Aeonic 1,000×12 (full cycle) | A4 vs A3 | 1.438× Linux / 1.610× Windows / 1.506× hosted macOS |

Full tables, raw evidence and platform limits are in [docs/performance.md](docs/performance.md).

### Added and changed

- Version `0.2.0` → `1.0.0`.
- CLI: `--backend {auto,python,cpp}`, `--device {auto,cpu,cuda,mps}`, `--torch-threads`, `--native-threads`, `--native-min-rows`, `--compute-only`, `--fast-forward`, `--no-interactive`, `--llm-backend`, `--llm-device`, `--gguf-model`, `--mlx-model`, `--seed`, `--output-dir`, `--smoke-test`.
- `--backend auto` prefers the native core, warns and falls back when it is missing; `cpp` fails clearly; `python` keeps the reference path.
- Defaults: `--torch-threads 1`, `--native-threads 1`, `--native-min-rows 4096`.
- `pip install ".[test]"` builds the native core; installing only `requirements.txt` does not.
- The new adapter `_tiemu_ffi` loads `_tiemu_core`; the legacy `_tiemu_native` extension is never selected.
- Outputs use atomic replacement; the default directory is `runs/`.

### Compatibility and upgrade notes

- **Snapshots are not resumable**: `world.json` is an inspection snapshot without optimizer, RNG or phase-internal state. `load` and `--load-save` refuse restoration and do not overwrite existing output. This is an intentional change from upstream.
- After changing C++ sources, rerun `pip install ".[test]"`; old `.so/.pyd` files are not refreshed automatically.
- Dependencies are pinned to NumPy `2.2.2` and PyTorch `2.12.0`; Python 3.11–3.13.
- Different devices, precision, quantization or LLM sampling do not guarantee identical long trajectories; short replays do not prove arbitrary long-term equivalence.
- The equivalence tests require complete Git history; run `git fetch --unshallow` on a shallow clone.

### Known limitations

- This is not a pure C++ rewrite: Python still owns model backends, some lifecycle decisions, online-training scheduling, console handling and snapshots.
- Multicore is limited to independent scoring; one thread is the default, four threads show no stable whole-phase gain, and interactions, sampling, vote reduction and training updates remain sequential.
- Some workloads/platforms regress: A2 aeonic `200×2` 0.944–0.958×; A3 macOS aeonic `200×2` 0.934×; A4 macOS aeonic `200×2`/`200×12` 0.908/0.902× and macOS inorganic 0.751×.
- The A5 neighbor-precomputation candidate measured 0.998× Linux, 1.023× Windows and 0.986× macOS on the actual inorganic phase; it is not enabled by default.
- The A4/A5 `1,000×12` full-cycle sample performs only 44 optimizer updates per optimizer and is not sustained-training throughput; the original reincarnator reference gap is retained.
- Windows/CUDA, Hyper-V GPU passthrough, later physical-M1 retests and ANE/Core ML are not validated; there is no Core ML backend.
- There is no multi-world, batch-experiment or parameter-search scheduler.

### Testing and validation

- 196 test functions; the A5 baseline default suite passes 613 tests on Linux/macOS and 601 with 12 skips on Windows.
- Standalone C/CTest, single/four-thread CLI and Linux CPU Docker pass.
- Frozen upstream `691c6e0` is replayed on the same platform; discrete decisions, RNG, weights and full Adam state are compared strictly; numeric tolerances are about `1e-10` for world values and `1e-6` for float32 networks.
- Release verification: `main` and `v1.0.0` point to the same commit, and the Release body matches this file.

### Documentation and license

- [README](README.md) · [Chinese guide](readme/readme_zh-cn.md) · [Performance and equivalence](docs/performance.md) · [Design documents and migration log](docs/design/) · [World parameter guide](readme/Guide.md) · [Logging](readme/clog_zh-cn.md)
- MIT License; original project and contributions: YezQiu, HeyCrab3, AvalonRuFae, MoonofBridge24, Akanyi. Logging uses [pyclog](https://github.com/Akanyi/pyclog).
