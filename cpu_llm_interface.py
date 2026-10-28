"""Local MLX/GGUF adapters, independent of the PyTorch training device."""

import logging
import platform
import re
from pathlib import Path

from config import config

logger = logging.getLogger("OmphalosLogger")


def _apple_silicon():
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _has_backend_marker(info, backend):
    """Accept actual llama.cpp backend headings/flags, not generic GPU support."""
    pattern = rf"\b(?:GGML_)?{re.escape(backend)}\s*(?::|=\s*(?:1|ON|TRUE)\b)"
    return re.search(pattern, info, flags=re.IGNORECASE) is not None


class CpuLlmInterface:
    """Keep the historical interface name while supporting native local backends.

    ``available`` is the readiness contract; callers must not inspect ``llm``.
    Automatic selection may disable the optional LLM after logged failures.
    Explicit backend/device requirements fail rather than silently downgrading.
    """

    def __init__(self, model_folder="models", settings=None):
        self.settings = dict(config["llm"] if settings is None else settings)
        self.available = False
        self.backend = "disabled"
        self.llm = None  # Compatibility for old code that only checked truthiness.
        self.device = "cpu"
        self._model_folder = Path(model_folder).expanduser()
        self._requested_backend = self.settings.get("backend", "auto")
        self._requested_device = self.settings.get("device", "auto")

        if not self.settings.get("enable_llm", True) or self._requested_backend == "disabled":
            logger.info("大模型功能已禁用。")
            return
        if self._requested_backend not in {"auto", "mlx", "llama_cpp"}:
            raise ValueError(f"未知 LLM 后端: {self._requested_backend}")
        if self._requested_device not in {"auto", "cpu", "cuda", "metal"}:
            raise ValueError(f"未知 LLM 设备: {self._requested_device}")

        candidates = [self._requested_backend]
        if self._requested_backend == "auto":
            prefer_mlx = _apple_silicon() and self._requested_device in {"auto", "metal"}
            candidates = ["mlx", "llama_cpp"] if prefer_mlx else ["llama_cpp"]

        last_error = None
        for candidate in candidates:
            try:
                if candidate == "mlx":
                    self._load_mlx()
                else:
                    self._load_gguf()
            except Exception as exc:
                last_error = exc
                logger.warning("LLM %s 初始化失败: %s", candidate.upper(), exc)
            else:
                self.backend = candidate
                self.available = True
                logger.info("LLM 后端: %s; 设备: %s", self.backend, self.device)
                return

        if self._requested_backend != "auto" or self._requested_device in {"cuda", "metal"}:
            raise RuntimeError(f"无法启用指定 LLM 后端/设备: {last_error}") from last_error
        logger.warning("所有自动 LLM 后端均不可用，继续运行模拟（LLM 已禁用）。")

    def _load_mlx(self):
        if not _apple_silicon():
            raise RuntimeError("本应用的 MLX 后端需要 Apple Silicon 原生 macOS。")
        if self._requested_device == "cuda":
            raise RuntimeError("本应用的 MLX 后端不支持 CUDA，请选择 llama_cpp。")
        path = Path(self.settings.get("mlx_model_path") or self._model_folder / "Qwen3-0.6B-MLX").expanduser()
        if not path.is_dir():
            raise RuntimeError(f"MLX 本地模型目录不存在: {path}；不会自动下载模型。")
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self.device = "cpu" if self._requested_device == "cpu" else "metal"
        if self.device == "metal" and not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU 不可用。")
        mx.set_default_device(mx.cpu if self.device == "cpu" else mx.gpu)
        if self.settings.get("seed") is not None:
            mx.random.seed(int(self.settings["seed"]))
        self._mlx_model, self._mlx_tokenizer = load(str(path))
        self._mlx_generate = generate
        self._mlx_sampler = make_sampler(temp=float(self.settings.get("temperature", 0.7)))
        self.llm = self._mlx_model

    def _gguf_layers(self, llama_cpp):
        requested = self._requested_device
        layers = int(self.settings.get("n_gpu_layers", self.settings.get("gpu_max_count", -1)))
        if layers < -1:
            raise ValueError("n_gpu_layers 必须是 -1 或非负整数。")
        if requested == "cpu":
            return 0, "cpu"
        system = platform.system()
        expected = "metal" if system == "Darwin" else "cuda" if system in {"Windows", "Linux"} else None
        if requested in {"cuda", "metal"} and requested != expected:
            raise RuntimeError(f"{requested.upper()} 不支持当前平台 {system}。")
        if layers == 0:
            if requested != "auto":
                raise RuntimeError(f"显式 {requested.upper()} 需要 n_gpu_layers 非零。")
            return 0, "cpu"

        # This is llama.cpp's own probe, not torch.cuda.is_available(). A generic
        # GPU flag can also mean Vulkan/RPC; require the matching build marker.
        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)()
        raw_info = getattr(llama_cpp, "llama_print_system_info", lambda: b"")()
        info = raw_info.decode("utf-8", errors="replace") if isinstance(raw_info, bytes) else str(raw_info)
        supported = bool(supports_gpu and expected and _has_backend_marker(info, expected))
        if supported:
            logger.info("llama.cpp 报告 %s 后端及 GPU offload 能力；请求卸载 %d 层。", expected.upper(), layers)
            return layers, expected
        if requested != "auto":
            raise RuntimeError(f"无法确认 llama.cpp 的 {requested.upper()} GPU 能力；需要对应构建和可用设备。")
        logger.warning("llama.cpp 未确认当前平台 GPU 后端，自动使用 CPU。")
        return 0, "cpu"

    def _load_gguf(self):
        configured_path = self.settings.get("gguf_model_path")
        path = Path(configured_path).expanduser() if configured_path else self._model_folder / self.settings.get("model_name", "Qwen3-0.6B-Q8_0.gguf")
        if not path.is_file():
            raise RuntimeError(f"GGUF 本地模型不存在: {path}；不会自动下载模型。")
        import llama_cpp

        layers, device = self._gguf_layers(llama_cpp)
        kwargs = {"model_path": str(path), "n_ctx": 4096, "n_gpu_layers": layers, "verbose": False}
        if self.settings.get("seed") is not None:
            kwargs["seed"] = int(self.settings["seed"])
        try:
            self.llm = llama_cpp.Llama(**kwargs)
        except Exception:
            if layers == 0 or self._requested_backend != "auto" or self._requested_device != "auto":
                raise
            logger.warning("llama.cpp GPU 加载失败，自动重试 CPU。", exc_info=True)
            kwargs["n_gpu_layers"] = 0
            self.llm = llama_cpp.Llama(**kwargs)
            device = "cpu"
        self.device = device

    def generate_response(self, prompt, max_tokens=120):
        if not self.available:
            return "错误：模型未能成功加载，无法生成响应。"
        thinking = bool(self.settings.get("enable_thinking", False))
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " + ("/think" if thinking else "/no_think")},
            {"role": "user", "content": prompt},
        ]
        if self.backend == "mlx":
            formatted = self._mlx_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
            )
            return self._mlx_generate(
                self._mlx_model, self._mlx_tokenizer, prompt=formatted,
                max_tokens=max_tokens, sampler=self._mlx_sampler, verbose=False,
            ).strip()
        output = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=float(self.settings.get("temperature", 0.7)),
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return (output["choices"][0]["message"].get("content") or "").strip()
