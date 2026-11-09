# 翁法罗斯参数配置指南

想让这个世界换一种活法，可以从这些参数开始。参数都在 `config.py` 里；改之前先看清楚它控制什么、调大调小会怎样。

## 零、大模型（LLM）配置 (`llm`)

这一节只影响叙事和少数决策。不需要的话，启动时加 `--disable-llm` 就行。

- `n_gpu_layers`
  - 交给 llama.cpp 放到 GPU 上的层数。`-1` 表示尽量全部卸载，纯 CPU 设为 `0`。旧配置里的 `gpu_max_count` 仍会作为回退值读取。

- `backend`
  - `auto` / `mlx` / `llama_cpp` / `disabled`。

- `device`
  - `auto` / `cpu` / `cuda` / `metal`。

- `mlx_model_path` / `gguf_model_path`
  - 本地模型目录或文件路径。程序不会自动下载模型。

- `temperature`
  - 采样温度。调高更发散，调低更稳定。

- `enable_llm`
  - 总开关。只想看纯演化就关掉。

- `kaoselanna_llm_enabled`
  - 白厄是否使用大模型做决策。

- `model_name`
  - GGUF 文件名，需要放在 `models` 目录里。

- `enable_thinking`
  - 只对 DeepSeek、QwQ 这类支持思考的模型有意义，默认关闭。

## 一、日志（Log）配置 (`log`)

- `enable`
  - 是否写日志文件。

- `file_name`
  - 日志文件名，自动带时间戳，不同次运行不会混在一起。

## 二、核心模拟参数 (`simulation`)

这一节改动影响最大，建议一次只调一两个值。

- `bard_frequency` / `laertes_frequency`
  - “吟游诗人”和“来古士”出现的频率，单位是世代。设为 `0` 禁用。

- `num_initial_entities`
  - 开局是热闹还是冷清，就看这个数了。

- `population_soft_cap` / `population_hard_cap`
  - 软上限以下会继续增长，超过硬上限则强制淘汰。

- `growth_factor`
  - 人口增长速度。数值越大，每代新生实体越多。

- `mutation_rate`
  - 基础突变率，决定新生实体继承演化蓝图时产生多大随机变化。

- `culling_strength`
  - 调得越高，弱势实体就越难存活。

- `encounter_similarity`
  - 实体倾向和评分多接近的对手交互。值越小，越容易碰上旗鼓相当的对手。

- `purity_factor`
  - 强化实体最主要的泰坦亲和度，同时削弱其他亲和度。

- `elite_selection_percentile`
  - 每代取评分排名前 N% 的精英，用来训练全局演化蓝图。

- `aeonic_event_prob`
  - 翁法罗斯事件的发生概率，例如大肃正、概念奇点。

- `target_avg_score` / `norm_adjustment_strength`
  - 一套宏观调控：尝试把实体平均分稳定在 `target_avg_score` 附近，`norm_adjustment_strength` 是调控力度。

## 三、模拟阶段结束点 (`simulation_phases`)

这些值划分世界历史阶段，通常不需要改。

- `INORGANIC_PHASE_END`: 无机推演阶段结束点。
- `ORGANIC_PHASE_END`: 有机测算阶段结束点。
- `HUMAN_PHASE_END`: 前黄金裔时代结束点，也是永劫回归可能开始的时刻。
- `TOTAL_SIMULATION_END`: 整个模拟的终点。

## 四、多样性干预系统 (`diversity_intervention`)

- `low_threshold` / `critical_threshold`
  - 多样性指数的低警告阈值和严重危机阈值。命途数量少到一定程度时触发干预。

- `base_mutation_rate`
  - 本系统管理的基础突变率，会根据多样性动态调整。

- 干预措施参数：
  - `schism_ratio`: 分裂事件中强制改变立场的人数比例。
  - `outsider_injection_count`: 天外来客事件注入的新实体数量。
  - `conformity_plague_strength`: 瘟疫事件中对主流思想追随者的评分惩罚系数。
  - `zeitgeist_suppression_factor`: 思潮抑制事件中削弱主流思想影响力的强度。
