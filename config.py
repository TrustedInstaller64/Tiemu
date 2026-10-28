from datetime import datetime

# 配置
config = {
    # 大模型
    "llm": {
        "n_gpu_layers": -1, # llama.cpp GPU 卸载层数；CPU 模式固定为0
        "backend": "auto", # auto / mlx / llama_cpp / disabled
        "device": "auto", # auto / cpu / cuda / metal
        "mlx_model_path": "models/Qwen3-0.6B-MLX",
        "gguf_model_path": None,
        "temperature": 0.7,
        "enable_llm": True, # 是否启用大模型
        "kaoselanna_llm_enabled": True, # 白厄的自我意识
        "model_name": "Qwen3-0.6B-Q8_0.gguf", # 模型名称
        "enable_thinking": False # 仅DeepSeek，QwQ等需要启用（不推荐）启用深度思考
    },
    "runtime": {
        "compute_backend": "auto", # auto / cpp / python
        "torch_device": "cpu",    # 当前单样本小网络优先 CPU；可选 auto/cuda/mps
        "torch_threads": 1,
        "seed": None,
        "interactive": None,      # None: 仅在真实终端启用
        "output_dir": "runs",
    },
    # 日志
    "log": {
        "enable": True, # 是否启用日志
        "file_name": 'simulation-{0}.clog'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), # 日志文件名
    },
    # 模拟参数
    "simulation": {
        "bard_frequency": 0,
        "laertes_frequency": 0,
        "num_initial_entities": 1000,
        "golden_one_cap": 12,
        "population_soft_cap": 1000,
        "population_hard_cap": 2000,
        "growth_factor": 0.35,
        "mutation_rate": 0.25,
        "culling_strength": 0.85,
        "encounter_similarity": 0.35,
        "purity_factor": 0.01,
        "initial_rl_lr": 0.005,
        "golden_one_reversion_prob": 0.1,
        "elite_selection_percentile": 80,
        "aeonic_event_prob": 0.05,
        "initial_max_affinity_norm": 10000.0,
        "target_avg_score": 15.0,
        "norm_adjustment_strength": 0.05,
        
        "elites_to_keep_in_cycle": 10 # 永劫回归时，保留的旧世界精英数量
    },
    # 模拟阶段结束点
    "simulation_phases": {
        "INORGANIC_PHASE_END": 50121,
        "ORGANIC_PHASE_END": 176199,
        "HUMAN_PHASE_END": 28371273,
        "TOTAL_SIMULATION_END": 335503360
    },
    # 多样性干预系统
    "diversity_intervention": {
        "low_threshold": 0.33,  # 低多样性警告阈值 (例如, 少于4个命途)
        "critical_threshold": 0.17, # 严重多样性危机阈值 (例如, 少于等于2个命途)
        "base_mutation_rate": 0.25, # 基础突变率，交由本系统管理
        
        # 干预措施参数
        "schism_ratio": 0.4, # 命途分裂时，被强制改变的比例
        "outsider_injection_count": 20, # 外来者的数量
        "conformity_plague_strength": 0.75, # 瘟疫的评分惩罚系数 (乘以该系数)
        "zeitgeist_suppression_factor": 0.5 # 思潮抑制的强度
    }
}
