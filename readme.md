# Tiemu

Native acceleration and platform support based on [YezQiu/Tiemu](https://github.com/YezQiu/Tiemu), baseline `691c6e0`.

[中文安装与运行](readme/readme_zh-cn.md) · [测量与等价性](docs/performance.md) · [世界参数指南](readme/Guide.md)

## Install and run

Use Python 3.12 and a C++17 compiler (MSVC Build Tools on Windows, Xcode Command Line Tools on macOS, GCC on Linux).

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[test]"
python -m main --backend cpp --device cpu --disable-llm --smoke-test
python -m pytest -q
```

Remove `--smoke-test` for the full simulation. Outputs go to `runs/`; use `--output-dir` to change this. `pip install .` builds the standalone C++ shared core; installing only `requirements.txt` does not.

C++ accelerates path arithmetic, purification, ordered interactions and their random draws, sequential grid updates, voting, opponent selection and path statistics. A resident World owns continuous state, stable entity IDs, membership reconciliation and batched newborn arithmetic through a C ABI, without pybind11. Python retains public entity mirrors, remaining lifecycle decisions and online training; current public fields are synchronized before native encounters. `--backend python` retains the reference path; `auto` reports a fallback if the native core is missing. See the [resident-world migration and validation](docs/a3-resident-world.md).

Small online networks default to CPU with one PyTorch thread, because MPS was slower for their current single-sample workload on the tested M1. `--device cuda`, `--device mps`, and `--torch-threads N` remain available. LLM device selection is independent: Apple Silicon can use local MLX/Metal, Windows/Linux can use an appropriate CUDA-enabled llama.cpp build, and CPU-only operation needs no GPU.

## Docker CPU

```sh
docker compose build
docker compose up -d
docker compose logs -f
docker compose stop
```

The default image disables LLM, runs as a non-root user, and stores outputs in the `tiemu-data` volume. It works without GPU passthrough; see the Chinese guide for optional CPU GGUF/model mounts and smoke checks. A Hyper-V guest does not automatically inherit the host GPU.

## Console and output

With an attached terminal, press `p` to pause at the next safe point. Commands include `c`, `n`, `status`, `top`, `p <name|baie>`, `zeitgeist`, `blueprint`, `set <parameter> <value>`, `save <path.json>`, `stop`, and `help`. `n` advances one generation or one organic training step. Linux/macOS use POSIX terminal input; Windows uses its console API. Non-TTY input is automatically disabled; `--no-interactive` makes this explicit.

Normal completion, `stop`, Ctrl+C and SIGTERM export `world.json` and CPU-loadable policy weights using atomic replacement. A JSON snapshot lacks optimizer, RNG and phase state and **cannot resume training**. `load` and `--load-save` therefore refuse restoration instead of silently restarting or overwriting a world. Files saved during an interrupted early phase are inspection snapshots, not checkpoints.

GPU/CPU arithmetic and independently sampled or quantized LLMs are not guaranteed to produce identical long trajectories or text. See [validation boundaries and measured speedups](docs/performance.md).

## Thanks

Original contributors: HeyCrab3, AvalonRuFae, MoonofBridge24, Akanyi.

Logging uses [pyclog](https://github.com/Akanyi/pyclog); see [logging guide](readme/clog_en.md).
