"""Backend boundary tests: no real model load or accelerator is required."""

import builtins
import importlib
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from cpu_llm_interface import CpuLlmInterface


def settings(**overrides):
    values = {
        "enable_llm": True,
        "backend": "auto",
        "model_name": "test.gguf",
        "device": "auto",
        "n_gpu_layers": -1,
        "temperature": 0.7,
        "enable_thinking": False,
        "seed": None,
    }
    values.update(overrides)
    return values


def platform(monkeypatch, system, machine="x86_64"):
    import platform as platform_module

    monkeypatch.setattr(platform_module, "system", lambda: system)
    monkeypatch.setattr(platform_module, "machine", lambda: machine)


def fake_llama(monkeypatch, gpu=False, info=b"CPU : AVX = 1", fail_gpu=False):
    module = ModuleType("llama_cpp")
    module.llama_supports_gpu_offload = lambda: gpu
    module.llama_print_system_info = lambda: info
    calls = {"loads": [], "chats": []}

    class Llama:
        def __init__(self, **kwargs):
            assert "enable_thinking" not in kwargs
            calls["loads"].append(kwargs)
            if fail_gpu and kwargs["n_gpu_layers"] != 0:
                raise RuntimeError("GPU out of memory")

        def create_chat_completion(self, **kwargs):
            calls["chats"].append(kwargs)
            return {"choices": [{"message": {"content": '  {"action":"谈判"}  '}}]}

    module.Llama = Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return calls


def fake_mlx(monkeypatch, metal=True):
    core = ModuleType("mlx.core")
    core.gpu, core.cpu = "gpu", "cpu"
    core.metal = SimpleNamespace(is_available=lambda: metal)
    calls = {"loads": [], "prompts": [], "samplers": [], "generations": [], "seeds": []}
    core.set_default_device = lambda device: calls.update(device=device)
    core.random = SimpleNamespace(seed=lambda seed: calls["seeds"].append(seed))
    mlx = ModuleType("mlx")
    mlx.core = core
    lm = ModuleType("mlx_lm")
    samples = ModuleType("mlx_lm.sample_utils")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["prompts"].append((messages, kwargs))
            return "templated chat"

    def load(path):
        calls["loads"].append(path)
        return object(), Tokenizer()

    def make_sampler(*, temp):
        calls["samplers"].append(temp)
        return "configured sampler"

    def generate(model, tokenizer, *, prompt, max_tokens, sampler, verbose):
        calls["generations"].append((prompt, max_tokens, sampler, verbose))
        return "  本地回答  "

    lm.load, lm.generate = load, generate
    samples.make_sampler = make_sampler
    for name, module in (("mlx", mlx), ("mlx.core", core), ("mlx_lm", lm), ("mlx_lm.sample_utils", samples)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.mark.parametrize("setting", [{"enable_llm": False}, {"backend": "disabled"}])
def test_disabled_does_not_import_model_packages(monkeypatch, setting):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("mlx", "llama_cpp"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    interface = CpuLlmInterface(settings=settings(**setting))
    assert not interface.available
    assert interface.backend == "disabled"
    assert "无法生成" in interface.generate_response("hello")


def test_cpu_gguf_forces_zero_gpu_layers_and_keeps_chat_contract(monkeypatch, tmp_path):
    platform(monkeypatch, "Windows")
    model = tmp_path / "test.gguf"
    model.touch()
    calls = fake_llama(monkeypatch, gpu=True, info=b"CUDA : ARCHS = 750")
    interface = CpuLlmInterface(tmp_path, settings=settings(backend="llama_cpp", device="cpu", seed=17))
    assert interface.available and interface.backend == "llama_cpp"
    assert calls["loads"][0]["n_gpu_layers"] == 0
    assert calls["loads"][0]["seed"] == 17
    assert interface.generate_response("hello", max_tokens=31) == '{"action":"谈判"}'
    chat = calls["chats"][0]
    assert chat["max_tokens"] == 31 and chat["temperature"] == 0.7
    assert chat["messages"][-1] == {"role": "user", "content": "hello"}
    assert "/no_think" in chat["messages"][0]["content"]
    assert chat["stop"] == ["<|im_end|>", "<|endoftext|>"]


@pytest.mark.parametrize("info", [b"CUDA : ARCHS = 750", b"GGML_CUDA=1", b"CUDA = 1 | CPU : AVX = 1"])
def test_cuda_gguf_uses_its_own_backend_capability(monkeypatch, tmp_path, info):
    platform(monkeypatch, "Windows")
    (tmp_path / "test.gguf").touch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = fake_llama(monkeypatch, gpu=True, info=info)
    interface = CpuLlmInterface(tmp_path, settings=settings(backend="llama_cpp", device="cuda"))
    assert interface.available
    assert calls["loads"][0]["n_gpu_layers"] == -1


@pytest.mark.parametrize("info", [b"Vulkan : devices = 1", b"CUDA = 0", b"GGML_CUDA=0", b"CPU : AVX = 1"])
def test_generic_gpu_support_is_not_cuda_support(monkeypatch, tmp_path, info):
    platform(monkeypatch, "Windows")
    (tmp_path / "test.gguf").touch()
    calls = fake_llama(monkeypatch, gpu=True, info=info)
    with pytest.raises(RuntimeError, match="CUDA"):
        CpuLlmInterface(tmp_path, settings=settings(device="cuda"))
    assert calls["loads"] == []


@pytest.mark.parametrize("system,device,info", [("Linux", "metal", b"Metal : EMBED_LIBRARY = 1"), ("Darwin", "cuda", b"CUDA : ARCHS = 750")])
def test_explicit_gpu_rejects_wrong_platform(monkeypatch, tmp_path, system, device, info):
    platform(monkeypatch, system)
    (tmp_path / "test.gguf").touch()
    fake_llama(monkeypatch, gpu=True, info=info)
    with pytest.raises(RuntimeError):
        CpuLlmInterface(tmp_path, settings=settings(backend="llama_cpp", device=device))


def test_auto_gguf_uses_cpu_when_gpu_backend_is_ambiguous(monkeypatch, tmp_path, caplog):
    platform(monkeypatch, "Linux")
    (tmp_path / "test.gguf").touch()
    calls = fake_llama(monkeypatch, gpu=True, info=b"Vulkan : devices = 1")
    interface = CpuLlmInterface(tmp_path, settings=settings())
    assert interface.available and calls["loads"][0]["n_gpu_layers"] == 0
    assert "CPU" in caplog.text


def test_auto_gpu_allocation_failure_retries_cpu(monkeypatch, tmp_path, caplog):
    platform(monkeypatch, "Windows")
    (tmp_path / "test.gguf").touch()
    calls = fake_llama(monkeypatch, gpu=True, info=b"CUDA : ARCHS = 750", fail_gpu=True)
    interface = CpuLlmInterface(tmp_path, settings=settings())
    assert interface.available
    assert [call["n_gpu_layers"] for call in calls["loads"]] == [-1, 0]
    assert "CPU" in caplog.text


@pytest.mark.parametrize("backend", ["auto", "llama_cpp"])
def test_missing_gguf_is_never_loaded_or_downloaded(monkeypatch, tmp_path, backend):
    platform(monkeypatch, "Linux")
    calls = fake_llama(monkeypatch)
    if backend == "auto":
        assert not CpuLlmInterface(tmp_path, settings=settings()).available
    else:
        with pytest.raises(RuntimeError, match="GGUF"):
            CpuLlmInterface(tmp_path, settings=settings(backend=backend))
    assert calls["loads"] == []


def test_auto_can_run_without_optional_llama_package(monkeypatch, tmp_path):
    platform(monkeypatch, "Linux")
    (tmp_path / "test.gguf").touch()
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    assert not CpuLlmInterface(tmp_path, settings=settings()).available
    with pytest.raises(RuntimeError, match="llama"):
        CpuLlmInterface(tmp_path, settings=settings(backend="llama_cpp"))


def test_auto_mac_selects_mlx_gpu_and_uses_chat_template_sampler(monkeypatch, tmp_path):
    platform(monkeypatch, "Darwin", "arm64")
    calls = fake_mlx(monkeypatch)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    interface = CpuLlmInterface(settings=settings(mlx_model_path=str(tmp_path), seed=5))
    assert interface.available and interface.backend == "mlx"
    assert calls["device"] == "gpu" and calls["seeds"] == [5]
    assert calls["loads"] == [str(tmp_path)]
    assert interface.generate_response("hello", max_tokens=22) == "本地回答"
    assert calls["prompts"][0][1] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert calls["samplers"] == [0.7]
    assert calls["generations"] == [("templated chat", 22, "configured sampler", False)]


def test_mlx_missing_directory_is_not_treated_as_a_hub_repository(monkeypatch, tmp_path):
    platform(monkeypatch, "Darwin", "arm64")
    calls = fake_mlx(monkeypatch)
    with pytest.raises(RuntimeError, match="MLX"):
        CpuLlmInterface(settings=settings(backend="mlx", mlx_model_path=str(tmp_path / "missing")))
    assert calls["loads"] == []


@pytest.mark.parametrize("failure", ["missing_model", "missing_package", "missing_metal"])
def test_auto_mac_falls_back_to_local_gguf(monkeypatch, tmp_path, failure, caplog):
    platform(monkeypatch, "Darwin", "arm64")
    mlx_path = tmp_path / "mlx"
    if failure != "missing_model":
        mlx_path.mkdir()
    fake_mlx(monkeypatch, metal=failure != "missing_metal")
    if failure == "missing_package":
        monkeypatch.setitem(sys.modules, "mlx_lm", None)
    (tmp_path / "test.gguf").touch()
    calls = fake_llama(monkeypatch, gpu=True, info=b"Metal : EMBED_LIBRARY = 1")
    interface = CpuLlmInterface(tmp_path, settings=settings(mlx_model_path=str(mlx_path)))
    assert interface.available and interface.backend == "llama_cpp"
    assert calls["loads"][0]["n_gpu_layers"] == -1
    assert "MLX" in caplog.text


def test_explicit_mlx_rejects_non_mac(monkeypatch, tmp_path):
    platform(monkeypatch, "Windows")
    calls = fake_mlx(monkeypatch)
    with pytest.raises(RuntimeError, match="MLX"):
        CpuLlmInterface(settings=settings(backend="mlx", mlx_model_path=str(tmp_path)))
    assert calls["loads"] == []


def test_auto_cpu_on_mac_does_not_load_mlx(monkeypatch, tmp_path):
    platform(monkeypatch, "Darwin", "arm64")
    (tmp_path / "test.gguf").touch()
    mlx_calls = fake_mlx(monkeypatch)
    llama_calls = fake_llama(monkeypatch, gpu=True, info=b"Metal : EMBED_LIBRARY = 1")
    interface = CpuLlmInterface(tmp_path, settings=settings(device="cpu", mlx_model_path=str(tmp_path)))
    assert interface.backend == "llama_cpp"
    assert mlx_calls["loads"] == [] and llama_calls["loads"][0]["n_gpu_layers"] == 0


@pytest.mark.parametrize("mps,cuda,want", [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")])
def test_torch_auto_device_capabilities(monkeypatch, mps, cuda, want):
    devices = importlib.import_module("device_utils")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert devices.resolve_torch_device().type == want
    assert devices.resolve_torch_device("cpu").type == "cpu"


@pytest.mark.parametrize("requested", ["mps", "cuda"])
def test_explicit_unavailable_torch_device_fails(monkeypatch, requested):
    devices = importlib.import_module("device_utils")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=requested):
        devices.resolve_torch_device(requested)


def test_unknown_torch_device_fails():
    devices = importlib.import_module("device_utils")
    with pytest.raises(ValueError):
        devices.resolve_torch_device("typo")


def test_explicit_cpu_does_not_probe_accelerator_runtimes(monkeypatch):
    devices = importlib.import_module("device_utils")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU selection probed CUDA"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: pytest.fail("CPU selection probed MPS"))
    assert devices.resolve_torch_device("cpu").type == "cpu"


def test_seed_repeats_cpu_streams_without_calling_unavailable_cuda(monkeypatch):
    devices = importlib.import_module("device_utils")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: pytest.fail("CUDA is unavailable"))
    devices.seed_everything(29)
    first = random.random(), np.random.rand(), torch.rand(3)
    devices.seed_everything(29)
    second = random.random(), np.random.rand(), torch.rand(3)
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_seed_initializes_available_accelerator_generators(monkeypatch):
    devices = importlib.import_module("device_utils")
    seeds = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: seeds.append(("cuda", seed)))
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: seeds.append(("mps", seed)))
    devices.seed_everything(7)
    assert seeds == [("cuda", 7), ("mps", 7)]


def test_none_seed_preserves_rng_state():
    devices = importlib.import_module("device_utils")
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    devices.seed_everything(None)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])
