# Tiemu

Tiemu 是一个以翁法罗斯为背景的演化模拟器。一次完整运行会依次走过无机网格、有机原型、人类文明（议会、泰坦、黄金裔）和永劫回归；最后阶段的白厄会在轮回里收集十二枚火种，并持续做在线训练。

本仓库从 [YezQiu/Tiemu](https://github.com/YezQiu/Tiemu) 的 `691c6e0` 出发，世界规则不动。主要改动是把 profile 里最热的路径搬进独立 C++17 核心，同时补齐跨平台构建和差分测试。v1.0.0 是严格等价的 A 线：状态、随机流和每次 optimizer 更新都与参考实现一致；允许轨迹分歧的 C 线实验不在这个版本里。

[English README](../README.md) · [发布说明](../RELEASE_NOTES.md) · [性能记录](../docs/performance.md) · [参数指南](Guide.md) · [日志说明](clog_zh-cn.md)

## 安装

需要 Python 3.11–3.13（测得最多的是 3.12）和 C++17 编译器。NumPy 固定 `2.2.2`，PyTorch 固定 `2.12.0`。

```sh
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[test]"
python -m main --backend cpp --device cpu --disable-llm --smoke-test
```

`_tiemu_core` 是 `pip install ".[test]"` 编译出来的。只装 `requirements.txt` 就只有 Python 后端：要么用 `--backend python`，要么单独构建核心：

```sh
cmake -S . -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone --config Release
ctest --test-dir build/standalone -C Release --output-on-failure
```

改过 `cpp/` 下的东西要重新安装，旧的共享库不会自己更新。

去掉 `--smoke-test` 才是完整阶段长度。输出默认在 `runs/`，可以用 `--output-dir` 改。

## 各层分工

Python 负责 CLI 和配置、阶段顺序、四个 PyTorch 网络及其在线训练、生命周期决策、控制台和快照。C++ 核心负责 profile 里真正花时间的部分：路径运算、有序候选索引、直方图、评分线程池、无机网格段、整代有序交互、常驻 World 状态和出生批处理。

两边通过 C ABI 连接，`_tiemu_ffi.py` 用 `ctypes` 加载 `_tiemu_core`。没有 pybind11，核心也不包含 Python 或 NumPy 头文件。

`--backend auto` 有原生核心就用，没有就警告并回退；`--backend cpp` 缺核心直接不启动；`--backend python` 留作参考路径。

## 线程

`--device` 管小网络。默认 CPU 单线程；在参考 M1 上，这种单样本前向比 MPS 快。

`--native-threads` 是独立实体评分的预算，包含调用线程，只有待评分实体达到 `--native-min-rows`（默认 4096）才启动辅助线程。默认 1 线程；四线程没有稳定的整阶段收益。交互、随机抽样、投票归约和 optimizer 更新都保持原顺序。

`--compute-only` 减少日志和展示等待，但不会跳过演算，也不会改变随机数消费。

## 暂停与输出

在安全点按 `p` 暂停。常用命令：`c`、`n`、`status`、`top`、`p <name|baie>`、`zeitgeist`、`blueprint`、`set <参数> <值>`、`save <文件.json>`、`stop`、`help`。没有 TTY 时键盘输入自动关闭。

正常结束、`stop`、Ctrl+C 和 SIGTERM 会用原子替换写出 `world.json` 和三个策略权重；有机训练完成时还会写 `organic_prototype_model.pth`。

`world.json` 是拿来看的快照，不是存档。它没有优化器、RNG 和阶段内部状态，所以 `load` 和 `--load-save` 会直接拒绝，而不是悄悄重开或覆盖世界。

## LLM（可选）

不会自动下载模型。`--disable-llm` 直接关掉叙事。

Apple Silicon 用 `--llm-backend mlx --mlx-model` 指向本地 MLX 目录；其他平台用 `--llm-backend llama_cpp --gguf-model` 指向本地 GGUF，CUDA 版的 llama.cpp 也可以。`auto` 在 macOS arm64 上先试 MLX、再试 GGUF，都不可用就记录原因并禁用。采样和量化会改变轨迹，不要指望不同后端吐出一样的文本。

## Docker

```sh
docker compose build
docker compose up -d
docker compose logs -f
docker compose stop
```

镜像只有 CPU 版，以 UID 10001 运行，默认关闭 LLM，输出保存在 `tiemu-data` 卷里。`docker build --build-arg WITH_LLM=1` 会加上 CPU GGUF 支持。没有配置 GPU 直通。

## 性能

成对的固定阶段样本，不是整个程序的数字，不同版本或场景的倍率也不能相乘。

| 场景 | 对照 | 结果 |
|---|---|---|
| 人类 1,000×6 | A2 vs A1 | 2.10–2.30× |
| 人类 10,000×3 | A2 vs A1 | 1.32–1.47× |
| 无机 10×10×10,000 | A2 vs A1 | 9.13–13.53× |
| 永劫 200×2（真实训练） | A2 vs A1 | 0.944–0.958×（更慢） |
| 人类 10,000×3 | A4 vs A3 | Linux 1.068× / Windows 1.140× / 托管 macOS 1.041× |
| 永劫 1,000×12（full cycle） | A4 vs A3 | 1.438× / 1.610× / 1.506× |

有些场景确实更慢：A2 永劫 200×2 是 0.944–0.958×，A3 macOS 永劫 200×2 是 0.934×，A4 macOS 永劫 200×2/200×12 是 0.908/0.902×，A4 macOS 无机是 0.751×。

1,000×12 的永劫样本每个 optimizer 只跑 44 次更新——原循环在换宿主后丢了轮回者引用——所以不能当成持续训练提速。四线程没有稳定的整阶段收益，默认仍是 1。

2026-10-28 在物理 M1 上测的 31.65×（人类）和 6.23×（无机）是相对原版 Python 循环的阶段数据，留作参考。

完整表格和原始证据见 [docs/performance.md](../docs/performance.md)。

## 测试

196 个测试函数。A5 基线里，参数化套件在 Linux/macOS 通过 613 项，Windows 通过 601 项、跳过 12 项。差分测试覆盖数值、有序交互、评分、出生、生命周期、常驻 World 和 C ABI；离散决策和 RNG 严格比较，世界值容差 `1e-10`，float32 网络约 `1e-6`。

等价性测试会从 Git 历史里回放上游 `691c6e0`，所以浅克隆要先 `git fetch --unshallow`。

## 目录

```
main.py, simulation.py        CLI 与主引擎
*_manager.py                  议会、多样性、人口、停滞、永劫循环
entities.py, models.py        实体与网络
native_*.py, _tiemu_ffi.py    原生适配器与 C ABI 桥接
cpp/                          C++17 核心
tests/                        pytest 套件、C 消费者、冻结夹具
benchmarks/                   计时与对照工具
docs/                         性能记录与设计文档
```

## 致谢

原项目与贡献者：YezQiu、HeyCrab3、AvalonRuFae、MoonofBridge24、Akanyi。日志使用 [pyclog](https://github.com/Akanyi/pyclog)。

MIT 许可证。非官方粉丝项目，游戏设定归其权利人所有。
