"""Runtime controls for one-world native workers; execute on remote CI only."""
import logging
import pytest
from config import config
from main import build_parser, configure_logging


def test_native_worker_budget_is_independent_of_torch_threads():
    args = build_parser().parse_args(['--native-threads', '4', '--native-min-rows', '128',
                                     '--torch-threads', '1', '--compute-only'])
    assert (args.native_threads, args.native_min_rows, args.torch_threads) == (4, 128, 1)
    assert args.compute_only is True


@pytest.mark.parametrize('option', ['--native-threads', '--native-min-rows'])
@pytest.mark.parametrize('value', ['0', '-1'])
def test_native_worker_budget_rejects_nonpositive_values(option, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([option, value])


def test_compute_logging_drops_info_output_but_keeps_argument_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(config['log'], 'enable', False)
    logger = logging.getLogger('OmphalosLogger')
    previous_level = logger.level
    calls = []
    class DescribedEntity:
        def __repr__(self):
            calls.append('path-cache-read')
            return 'verbose-entity'
    try:
        configure_logging(tmp_path, fast_forward=False, compute_only=True)
        logger.info(f'{DescribedEntity()}')
        logger.warning('important-warning')
        output = capsys.readouterr().out
        assert 'verbose-entity' not in output
        assert 'important-warning' in output
        assert calls == ['path-cache-read']
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_compute_mode_pause_commands_are_visible_and_resume_restores_quiet_logging(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from debugger import Debugger
    from simulation import AeonEvolution
    monkeypatch.setitem(config['log'], 'enable', False)
    logger = logging.getLogger('OmphalosLogger')
    previous_level = logger.level
    sim = object.__new__(AeonEvolution)
    sim.population = []
    sim.generation, sim.total_generations = 7, 100
    sim.reincarnator = None
    sim.population_manager = SimpleNamespace(mutation_rate=.25)
    sim.stagnation_manager = SimpleNamespace(long_term_stagnation_counter=0)
    debugger = Debugger(sim, interactive=False)
    sim.debugger = debugger
    class ScriptedConsole:
        interactive = True
        def __init__(self): self.characters = iter('help\nstatus\nc\n')
        def read_char(self, timeout=0): return next(self.characters)
        def write(self, text): pass
        def close(self): self.interactive = False
    debugger._console = ScriptedConsole()
    debugger.paused = True
    try:
        configure_logging(tmp_path, fast_forward=False, compute_only=True)
        assert sim._control_checkpoint()
        output = capsys.readouterr().out
        assert '模拟已暂停' in output
        assert '状态报告' in output
        assert '7/100' in output
        assert logger.level == logging.WARNING
        logger.info('hidden-after-resume')
        assert 'hidden-after-resume' not in capsys.readouterr().out
    finally:
        debugger.close()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(previous_level)
