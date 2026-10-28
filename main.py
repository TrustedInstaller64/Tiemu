"""Cross-platform CLI. Importing this module does not open files or start threads."""
import argparse
import logging
from pathlib import Path
import signal
import sys
from config import config

logger = logging.getLogger('OmphalosLogger')

def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('必须是大于 0 的整数')
    return number

def build_parser():
    parser = argparse.ArgumentParser(description='Tiemu 翁法罗斯演化模拟')
    parser.add_argument('--disable-llm', action='store_true', help='无需安装或加载 LLM 即可运行')
    parser.add_argument('--load-save', help='保留旧参数；JSON 快照目前不支持恢复运行')
    parser.add_argument('--fast-forward', action='store_true', help='减少控制台日志及展示等待')
    parser.add_argument('--no-interactive', action='store_true', help='不读取键盘，适用于 Docker/后台任务')
    parser.add_argument('--backend', choices=['auto', 'python', 'cpp'], default=config.get('runtime', {}).get('compute_backend', 'auto'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default=config.get('runtime', {}).get('torch_device', 'auto'))
    parser.add_argument('--torch-threads', type=positive_int, default=config.get('runtime', {}).get('torch_threads', 1), help='PyTorch CPU 运算线程数；当前小网络默认 1')
    parser.add_argument('--llm-backend', choices=['auto', 'mlx', 'llama_cpp', 'disabled'], default=config['llm'].get('backend', 'auto'))
    parser.add_argument('--llm-device', choices=['auto', 'cpu', 'cuda', 'metal'], default=config['llm'].get('device', 'auto'))
    parser.add_argument('--gguf-model', help='本地 GGUF 模型路径')
    parser.add_argument('--mlx-model', help='本地 MLX 模型目录')
    parser.add_argument('--seed', type=int, default=config.get('runtime', {}).get('seed'))
    parser.add_argument('--output-dir', default=config.get('runtime', {}).get('output_dir', 'runs'), help='快照、模型和日志输出目录')
    parser.add_argument('--smoke-test', action='store_true', help='缩短各阶段并使用 24 个实体做完整启动验证')
    return parser

def configure_logging(output_dir, fast_forward):
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    console.setLevel(logging.WARNING if fast_forward else logging.INFO)
    logger.addHandler(console)
    if config['log']['enable']:
        from pyclog import ClogFileHandler, constants
        file_handler = ClogFileHandler(str(output_dir/config['log']['file_name']), compression_code=constants.COMPRESSION_GZIP)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

def run_simple(args):
    import torch
    from native_backend import configure_backend
    from device_utils import seed_everything, resolve_torch_device
    from simulation import AeonEvolution

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, args.fast_forward)
    runtime = config.setdefault('runtime', {})
    runtime.update(torch_device=args.device, interactive=False if args.no_interactive else None,
                   output_dir=str(output_dir), seed=args.seed, torch_threads=args.torch_threads)
    torch.set_num_threads(args.torch_threads)
    config['llm'].update(backend=args.llm_backend, device=args.llm_device, seed=args.seed)
    if args.disable_llm or args.llm_backend == 'disabled':
        config['llm']['enable_llm'] = False
    if args.gguf_model:
        config['llm']['gguf_model_path'] = str(Path(args.gguf_model).expanduser().resolve())
    if args.mlx_model:
        config['llm']['mlx_model_path'] = str(Path(args.mlx_model).expanduser().resolve())
    if args.smoke_test:
        config['simulation_phases'].update(INORGANIC_PHASE_END=2, ORGANIC_PHASE_END=4,
                                           HUMAN_PHASE_END=8, TOTAL_SIMULATION_END=10)
        config['simulation'].update(num_initial_entities=24, population_soft_cap=24, population_hard_cap=48)

    sim = None
    exit_code = 0
    pending_signal = None
    previous_handlers = {}
    def stop_handler(signum, frame):
        nonlocal pending_signal
        pending_signal = signum
        if sim is not None:
            sim.debugger.request_stop(signum)

    try:
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            number = getattr(signal, name, None)
            if number is not None:
                previous_handlers[number] = signal.signal(number, stop_handler)
        native = configure_backend(args.backend)
        device = resolve_torch_device(args.device)
        if args.seed is not None:
            seed_everything(args.seed)
        logger.warning('计算后端: %s | 小网络设备: %s | 输出: %s', 'C++' if native is not None else 'Python', device, output_dir)
        kwargs = {k: v for k, v in config['simulation'].items() if k != 'elites_to_keep_in_cycle'}
        sim = AeonEvolution(**kwargs, kaoselanna_llm_enabled=config['llm']['kaoselanna_llm_enabled'],
                            fast_forward=args.fast_forward)
        if pending_signal is not None:
            sim.debugger.request_stop(pending_signal)
        sim.start(num_generations=config['simulation_phases']['TOTAL_SIMULATION_END'])
        if sim.debugger.stop_signal is not None:
            exit_code = 128 + sim.debugger.stop_signal
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        logger.exception('模拟运行失败')
        exit_code = 1
    finally:
        if sim is not None:
            try:
                if not sim.save_simulation_state(str(output_dir/'world.json')):
                    exit_code = 1
                sim.policy_saver.save_policy_models()
            except Exception:
                logger.exception('退出时保存失败')
                exit_code = 1
            finally:
                sim.debugger.close()
        for number, previous in previous_handlers.items():
            signal.signal(number, previous)
        logging.shutdown()
    return exit_code

def main(argv=None):
    # Windows redirected output otherwise inherits a legacy code page that may
    # not encode entity names. Keep logs/pipes UTF-8 on every platform.
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.load_save:
        parser.error('JSON 快照不含模型、优化器、随机数和阶段内部状态，当前不能恢复运行；已停止，未创建或覆盖输出文件。')
    return run_simple(args)

if __name__ == '__main__':
    raise SystemExit(main())
