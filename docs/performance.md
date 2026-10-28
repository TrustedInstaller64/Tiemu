# 性能与等价性记录（2026-10-28）

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
