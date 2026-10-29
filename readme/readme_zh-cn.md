# Tiemu 安装与运行

本分支基于 [YezQiu/Tiemu](https://github.com/YezQiu/Tiemu) 的 `691c6e0` 重做加速，保留原有世界规则。不是对 AMPHOREUS 方案的延续。

[世界参数指南](Guide.md) · [实测性能和验证边界](../docs/performance.md)

## 安装 C++ 核心

推荐 Python 3.12（包支持 3.11～3.13）。数值依赖固定为 NumPy 2.2.2 / PyTorch 2.12.0，以便复现测试。需要 C++17 编译器：Linux 使用 GCC/build-essential，macOS 安装 Xcode Command Line Tools，Windows 安装 Visual Studio Build Tools 的“使用 C++ 的桌面开发”。Mac 使用原生 arm64 Python。

```sh
git clone --branch codex/single-world-a git@github.com:TrustedInstaller64/Tiemu.git
cd Tiemu
python -m venv .venv
```

macOS/Linux 激活：`source .venv/bin/activate`；Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`。

```sh
python -m pip install --upgrade pip
python -m pip install ".[test]"
python -m main --backend cpp --device cpu --disable-llm --no-interactive --seed 42 --smoke-test
python -m pytest -q
```

`pip install .` 才会编译、安装原生扩展；仅安装 `requirements.txt` 不会构建 C++。默认不安装 LLM 框架，不下载模型。开发时修改 C++ 后需要重新安装项目。

去掉 `--smoke-test` 才是原参数的完整模拟。烟测会缩短阶段并使用 24 个初始实体，不能用于衡量完整模拟结果。完整模拟默认阶段很长，先做烟测。

```sh
python -m main --backend cpp --device cpu --torch-threads 1 --disable-llm --fast-forward --output-dir runs/cpu
```

`--fast-forward` 减少终端输出和展示等待；文件日志仍受 `config.py` 中 `log.enable` 控制，大量日志会增加 I/O 成本。要测计算性能，可在配置中关闭文件日志。`--backend python` 保留参考路径；`auto` 在扩展缺失时会明确提示并回退，`cpp` 则明确报错。

## A：单个世界的 C++ 批处理与多核

`codex/single-world-a` 是保留原规则的主线；`codex/experimental-c` 是独立的长期算法实验分支。目前没有多世界、批量实验或参数搜索调度器。

在允许负载的目标机器上安装/重建扩展后，可以启用同一世界内独立评分的并行计算：

```sh
python -m main --backend cpp --device cpu --torch-threads 1 --native-threads 4 --native-min-rows 4096 --compute-only --output-dir runs/single-world
```

此命令保持配置中的 LLM 设置；`--disable-llm` 只适合明确希望关闭 LLM 的场景，LLM 决策会影响演化结果。`--native-threads` 控制 C++ 独立实体评分的工作预算，包含调用线程，4 表示最多3个辅助线程；`--torch-threads` 控制 PyTorch，二者不能当作整个程序统一的线程上限。默认 C++ 1 线程，达到 `--native-min-rows`（默认4096）才启动辅助线程；小批次直接串行处理。它不会并行相互依赖的交互、随机抽样、投票求和或在线训练更新。

`--compute-only` 抑制 INFO 文件/控制台日志和有机阶段的展示等待，保留警告、错误，以及原日志参数求值中的随机数和缓存副作用。暂停后仍能查看命令输出，继续后恢复安静。它不会隐式开启 `--fast-forward` 的额外状态摘要，因为该摘要也可能读取懒缓存。

嵌入 Python 使用模拟器时，在结束后调用 `sim.close()`；直接使用 `BatchEvaluator` 时调用其 `close()`。原生调用释放 GIL，调用期间不得从其它线程修改或 resize 输入数组。

当前开发 Mac 暂时禁止本地高负荷；上述运行、构建和测试说明用于获准的目标机器，不能作为恢复本机运行的授权。C++ 源码改变后旧本地 `.so/.pyd` 不会自动更新，未重建前不要用旧扩展执行新分支。

## Apple Silicon：LLM 使用 MLX/Metal

```sh
python -m pip install ".[mlx]"
python -m main --backend cpp --device cpu --llm-backend mlx --llm-device metal --mlx-model /绝对路径/Qwen3-0.6B-MLX --output-dir runs/mac
```

必须先准备完整的本地 MLX 模型目录，包含模型配置、tokenizer 和权重；GGUF 文件不能作为 MLX 模型目录。程序不会隐式下载。默认自动选择时，原生 Apple Silicon macOS 优先尝试 MLX，再尝试 GGUF；全部不可用会记录原因并禁用可选 LLM。明确指定后端或 GPU 却不可用时会报错。

小型在线网络与 LLM 分开选设备。当前 M1 实测中，小网络 CPU 单线程比 MPS 快，所以默认 `--device cpu`；需要测试 Metal 小网络可传 `--device mps`。`--device auto` 是硬件可用性选择，并不是性能自动调优。当前 MLX 后端面向原生 Apple Silicon macOS，不在 Linux Docker 中启用。

**ANE：当前版本未实现 Core ML 后端。** MLX 使用 CPU/Metal GPU，不调用 ANE。固定模型可另行转换到 Core ML，并允许 CPU+ANE 推理；这不等于整个演化循环或在线 Adam 训练能移到 ANE，也不保证全部算子在 ANE 执行。需要针对实际转换模型验证输出、算子分配和功耗，不能仅依据设备名称声称省电。[Apple MLX 说明](https://developer.apple.com/videos/play/wwdc2025/315/)、[Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)。

若目的是同时正常使用电脑，可先用默认单线程 CPU 小网络，并以 `nice -n 10 python -m main ...` 降低 macOS/Linux 进程调度优先级；LLM 可继续使用 MLX，或关闭 LLM 进一步减少争用。这是资源控制建议，不是能耗测试结论。

## Windows 原生与 CUDA

Windows 无需 Docker，安装和运行命令与上面一致。CPU 模式不需要 CUDA。

小网络要启用 CUDA，先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装匹配显卡/驱动的 CUDA 版，再使用 `--device cuda`。默认 CPU 版包不能因为电脑有显卡就自动变成 CUDA 版。

LLM 的 llama.cpp 构建与 PyTorch 独立。安装 CUDA Toolkit 和 C++ 工具链后，可在 PowerShell 构建 CUDA 后端：

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
python -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python "llama-cpp-python>=0.3.14,<0.4"
Remove-Item Env:CMAKE_ARGS
python -m main --backend cpp --device cpu --llm-backend llama_cpp --llm-device cuda --gguf-model C:\models\Qwen3-0.6B-Q8_0.gguf
```

详见 [llama-cpp-python 官方安装说明](https://github.com/abetlen/llama-cpp-python#installation)。启动日志会记录请求的后端和层数；明确 GPU 配置不会悄悄回退到 CPU。`config.py` 的 `llm.n_gpu_layers=-1` 请求卸载全部可卸载层，显存不足时需要调小或改用 CPU。

## Hyper-V Linux + Docker（默认 CPU）

在 Linux 虚拟机里安装 Docker Engine 和 Compose 后，在项目目录执行：

```sh
docker compose build
docker compose run --rm tiemu --no-interactive --disable-llm --device cpu --backend cpp --seed 42 --smoke-test --output-dir /data
docker compose up -d
docker compose logs -f
docker compose stop
```

默认镜像为 CPU PyTorch、C++ 核心、不加载 LLM，以 UID 10001 运行；不要求虚拟机看到物理 GPU。命名卷 `tiemu-data` 持久保存 `/data` 中的快照、权重和日志。停止容器留出 60 秒保存时间，不要通过 `docker compose down -v` 删除需要保留的数据。若改用宿主绑定目录，须让 UID 10001 有写权限。

需要 CPU GGUF 时单独构建，并只读挂载模型：

```sh
docker build --build-arg WITH_LLM=1 -t tiemu:cpu-llm .
docker run --rm --init --stop-timeout 60 -v tiemu-output:/data -v /宿主模型目录:/models:ro tiemu:cpu-llm --no-interactive --device cpu --backend cpp --llm-backend llama_cpp --llm-device cpu --gguf-model /models/Qwen3-0.6B-Q8_0.gguf --output-dir /data
```

Docker 默认 CMD 禁用 LLM；上面的完整命令覆盖 CMD 后才启用 CPU LLM。宿主有 NVIDIA 显卡不意味着 Hyper-V Linux 已配置 GPU 直通；本镜像不配置直通，也不包含 CUDA。

## 暂停、输出与恢复边界

有真实终端时按 `p`，到下一个安全点暂停；输入 `c` 继续、`n` 单步、`status`/`top` 查看、`save 文件.json` 导出快照、`stop` 保存并退出。`help` 显示其余查询及参数修改命令。`n` 在有机网络训练期间表示一个训练步，其他阶段表示一代。

macOS/Linux 使用 POSIX TTY，Windows 使用控制台接口。重定向、后台或 Docker 无 TTY 时自动关闭键盘输入，也可显式使用 `--no-interactive`。Ctrl+C、SIGTERM 和 Windows Ctrl+Break 在安全点退出；强杀进程无法执行保存。

`--output-dir` 默认 `runs`。正常完成/受控停止会原子写入 `world.json`、`blueprint_policy.pth`、`baie_action_policy.pth`、`baie_value_policy.pth`；完成有机训练时还保存 `organic_prototype_model.pth`。权重转到 CPU 保存，便于跨设备加载。

**JSON 是供检查的世界快照，不是可恢复的完整训练断点。** 它缺少优化器、随机状态、阶段内部数据等。旧版宣称可以恢复，但实际入口会重跑早期阶段；本版本的 `load` / `--load-save` 明确拒绝恢复，防止世代回退或覆盖现有输出。早期阶段停止时的快照同样不能续跑。

## 验证

等价性测试需完整 Git 历史，以导出固定提交的原版并在同平台运行；浅克隆请先执行 `git fetch --unshallow`。测试包括 C++ 与参考核差分、冻结原版的种群/随机状态/训练回放、POSIX PTY、无终端 CLI、SIGTERM 保存以及设备选择。具体实测平台、速度和未验证范围见 [性能记录](../docs/performance.md)。不同设备、精度、模型量化和 LLM 采样不保证逐位相同，长程随机演化也不能以短回放证明完全一致。

## 致谢

HeyCrab3、AvalonRuFae、MoonofBridge24、Akanyi，以及原项目作者和贡献者。日志说明见 [clog_zh-cn.md](clog_zh-cn.md)。
