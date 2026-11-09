# Tiemu

Tiemu 是一个以《崩坏：星穹铁道》翁法罗斯设定为背景的演化模拟器。一次运行依次经历四个阶段：无机 10×10 网格、有机原型网络、带议会与泰坦事件的人类/命途行者文明，以及包含在线强化学习的永劫回归。

本项目基于 [YezQiu/Tiemu](https://github.com/YezQiu/Tiemu) 的 `691c6e0`，保留原有世界规则，主要新增独立 C++17 数值核心、跨平台打包和差分验证。

[English README](../README.md) · [发布说明](../RELEASE_NOTES.md) · [性能与等价性](../docs/performance.md) · [世界参数指南](Guide.md) · [日志说明](clog_zh-cn.md)

## v1.0.0 概览

- Python 3.11–3.13 编排层加独立 C++17 核心。
- 两层通过 C ABI 和薄层 `ctypes` 适配器连接，不使用 pybind11。
- C++ 常驻 World 持有连续数值状态、稳定实体 ID、人口顺序、成员对账和出生数值批处理。
- 保留有序语义：交互顺序、随机数消费、旧缓存行为和每次在线 optimizer 更新。
- Linux、Windows、macOS CI；独立 CMake/CTest；Linux CPU Docker 镜像。
- 196 个测试函数，包含冻结上游回放和严格的 RNG/离散决策比较。

## 架构

本项目是 Python 应用加原生数值核心，不是纯 C++ 程序，也不把 Python 嵌入 C++。

| 层 | 职责 |
|---|---|
| Python | CLI 与配置、阶段编排、PyTorch 模型定义与在线训练、生命周期决策、控制台、快照、可选 LLM 后端 |
| C++17 核心（`cpp/`） | 数值核、有序候选索引、直方图、评分线程池、无机网格段、有序实体交互、常驻世界状态、出生数值批处理 |
| 桥接层 | C ABI 共享库 `_tiemu_core`，由 `_tiemu_ffi` 通过 `ctypes` 加载；不使用 Python C API、NumPy 头文件或 pybind11 |

后端选择：

- `--backend auto`（默认）：原生核心可用时使用；缺失时记录警告并回退 Python。
- `--backend cpp`：要求原生核心；缺失时明确报错。
- `--backend python`：保留参考路径，用于对照和非标准输入。

## 环境要求

- Python 3.11–3.13（推荐 3.12）。
- C++17 编译器：Linux 使用 GCC，macOS 安装 Xcode Command Line Tools，Windows 安装 Visual Studio Build Tools 的“使用 C++ 的桌面开发”工作负载。
- 固定数值依赖：NumPy `2.2.2`、PyTorch `2.12.0`。
- 可选 LLM 后端：Apple Silicon 上的 MLX，或带本地 GGUF 模型的 `llama-cpp-python`。程序不会隐式下载模型。

## 安装

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[test]"
```

`pip install ".[test]"` 会编译并安装 C++ 共享核心。只安装 `requirements.txt` 不会构建核心；此时应使用 `--backend python`，或单独构建核心。修改 C++ 源码后需要重新安装项目，旧 `.so/.pyd` 不会自动更新。

核心也可以脱离 Python 独立构建：

```sh
cmake -S . -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone --config Release
ctest --test-dir build/standalone -C Release --output-on-failure
```

## 快速开始

```sh
# 端到端烟测（24 个实体、缩短阶段）
python -m main --backend cpp --device cpu --disable-llm --smoke-test

# 完整模拟
python -m main --backend cpp --device cpu --torch-threads 1 --disable-llm --fast-forward --output-dir runs/cpu

# 测试（需要完整 Git 历史以回放冻结上游）
python -m pytest -q
```

去掉 `--smoke-test` 才是完整阶段长度。输出默认写入 `runs/`，可用 `--output-dir` 修改。

## 后端与线程

- `--device {auto,cpu,cuda,mps}` 选择小型在线网络的设备。参考 M1 上这些小网络的单样本负载以 CPU 单线程最快，因此默认 `cpu`。
- `--torch-threads N` 设置 PyTorch CPU 线程数。
- `--native-threads N` 设置 C++ 独立实体评分的线程预算。预算包含调用线程；只有待评分实体达到 `--native-min-rows` 时才启动辅助线程。默认分别为 `1` 和 `4096`。
- `--compute-only` 抑制 INFO 日志和展示等待，同时保留演算、随机数消费和日志参数求值副作用。

只有独立评分并行执行。有序交互、随机抽样、投票归约和在线 optimizer 更新保持串行。

## 控制台、输出与快照

有真实终端时按 `p` 在安全点暂停。命令包括 `c`、`n`、`status`、`top`、`p <name|baie>`、`zeitgeist`、`blueprint`、`set <parameter> <value>`、`save <path.json>`、`stop` 和 `help`。非 TTY 输入会自动关闭；可用 `--no-interactive` 显式指定。

正常完成、`stop`、Ctrl+C 和 SIGTERM 会以原子替换方式写入 `world.json` 和可移植到 CPU 的策略权重。完成有机训练时还会写入 `organic_prototype_model.pth`。

`world.json` 是供检查的快照，不是可恢复的断点：它不包含优化器状态、随机数状态和阶段内部数据。`load` 与 `--load-save` 因此明确拒绝恢复，避免静默重启世界或覆盖现有输出。

## 可选 LLM 后端

LLM 叙事是可选功能，可用 `--disable-llm` 关闭。

- Apple Silicon：使用 MLX/Metal 和本地模型目录（`--llm-backend mlx --mlx-model /绝对路径/Qwen3-0.6B-MLX`）。
- Windows/Linux 或纯 CPU：使用 `llama-cpp-python` 和本地 GGUF 文件（`--llm-backend llama_cpp --gguf-model /绝对路径/model.gguf`），可选 CUDA。
- `auto` 在原生 Apple Silicon macOS 上优先尝试 MLX，其次 GGUF；全部不可用时记录原因并禁用 LLM。

模型不会隐式下载。独立采样或量化的 LLM 不保证产生相同的长程轨迹。

## Docker（CPU）

```sh
docker compose build
docker compose up -d
docker compose logs -f
docker compose stop
```

默认镜像以非 root 用户（UID 10001）运行、禁用 LLM，并把输出保存在 `tiemu-data` 命名卷中。它不要求 GPU 直通。需要 CPU GGUF 时单独构建：

```sh
docker build --build-arg WITH_LLM=1 -t tiemu:cpu-llm .
```

## 平台支持

- `native` CI 工作流覆盖 Linux、Windows、macOS：独立 CMake/CTest、纯 C ABI 消费者、`pip install`、pytest 和 CLI 烟测。
- Linux CPU Docker 在 CI 中构建并烟测。
- 历史 M1 测量覆盖 macOS C++/CPU、POSIX 控制台、MLX Metal 和 MPS 前反向。
- Windows 控制台逻辑使用模拟 API 测试。Windows/CUDA、Hyper-V GPU 直通和 ANE/Core ML 没有真机验收；当前没有 Core ML 后端。

## 性能摘要

以下是成对运行的固定阶段样本，不是完整程序或长期吞吐承诺。不同版本、不同场景的倍率不能相乘。托管 macOS 是虚拟 M1，不是物理硬件。

| 场景 | 对照 | 结果 |
|---|---|---|
| 人类 1,000×6 | A2 vs A1 | 2.10–2.30× |
| 人类 10,000×3 | A2 vs A1 | 1.32–1.47× |
| 无机 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| 永劫 200×2（真实训练） | A2 vs A1 | 0.944–0.958×（变慢） |
| 人类 10,000×3 | A4 vs A3 | Linux 1.068× / Windows 1.140× / 托管 macOS 1.041× |
| 永劫 1,000×12（full cycle） | A4 vs A3 | Linux 1.438× / Windows 1.610× / 托管 macOS 1.506× |

重要限制：

- A4/A5 的 `1,000×12` full-cycle 样本中，每个 optimizer 只执行 44 次更新，因为原循环在更换宿主后丢失了轮回者引用。不能把它称作持续多轮训练提速。
- 部分场景确实变慢：A2 永劫 `200×2` 为 0.944–0.958×；A3 macOS 永劫 `200×2` 为 0.934×；A4 macOS 永劫 `200×2`/`200×12` 为 0.908/0.902×，macOS 无机为 0.751×。
- 四个评分线程没有稳定的整阶段收益；默认仍为单线程。
- 2026-10-28 物理 M1 的阶段基准（人类 31.65×、无机 6.23×，相对原版 Python 循环）仅是历史阶段测量。

完整表格、原始证据和平台限制见 [docs/performance.md](../docs/performance.md)。

## 测试与等价性

- 196 个测试函数。参数化的默认套件在 Linux/macOS 通过 613 项，Windows 通过 601 项并跳过 12 项平台相关测试（A5 基线）。
- 差分测试覆盖世界数值、有序交互、评分、出生、生命周期、常驻 World 和纯 C ABI。
- 同一平台回放冻结上游 `691c6e0`；离散决策和全部 RNG 状态严格比较，数值使用紧容差（世界值约 `1e-10`，float32 网络输出约 `1e-6`）。
- 等价性测试需要完整 Git 历史。浅克隆请先执行 `git fetch --unshallow`。

## 目录结构

```
main.py, simulation.py        CLI 与主模拟引擎
*_manager.py                  议会、多样性、人口、停滞、永劫循环
entities.py, models.py        实体与 PyTorch 网络
native_*.py, _tiemu_ffi.py    原生适配器与 C ABI 桥接
cpp/                          独立 C++17 核心（CMake）
tests/                        pytest 套件、C ABI 消费者与冻结夹具
benchmarks/                   计时、剖析与对照工具
docs/                         性能记录与设计文档
readme/                       中文指南、参数指南与日志说明
```

## 文档索引

- [发布说明](../RELEASE_NOTES.md)
- [性能与等价性记录](../docs/performance.md)
- [设计文档与迁移日志](../docs/design/)
- [世界参数指南](Guide.md)
- [日志说明](clog_zh-cn.md)

## 致谢与许可

原项目与贡献者：YezQiu、HeyCrab3、AvalonRuFae、MoonofBridge24、Akanyi。日志使用 [pyclog](https://github.com/Akanyi/pyclog)。

本项目以 MIT 许可证发布；见 [LICENSE](../LICENSE)。这是非官方粉丝项目；游戏名称与设定归其权利人所有。
