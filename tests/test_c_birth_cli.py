"""Cloud-only contracts for the C2 opt-in CLI."""
import importlib

import pytest

from config import config
from main import build_parser


native = importlib.import_module('_tiemu_ffi')
pytestmark = pytest.mark.xfail(
    not hasattr(native.BatchEvaluator, 'newborn_affinities'),
    raises=AttributeError, strict=True,
    reason='C2 RED: newborn C ABI is not implemented')


def test_c2_is_opt_in_and_independent_of_c1():
    getattr(native.BatchEvaluator, 'newborn_affinities')
    assert config['runtime']['experimental_c_newborns'] is False
    baseline = build_parser().parse_args([])
    assert baseline.experimental_c_newborns is False
    enabled = build_parser().parse_args(['--experimental-c-newborns', '--native-threads', '4'])
    assert enabled.experimental_c_newborns is True
    assert enabled.experimental_c_scoring is False
    assert enabled.native_threads == 4


def test_c1_and_c2_can_share_the_explicit_worker_budget():
    getattr(native.BatchEvaluator, 'newborn_affinities')
    args = build_parser().parse_args(['--experimental-c-scoring', '--experimental-c-newborns',
                                    '--native-threads', '2', '--native-min-rows', '1'])
    assert args.experimental_c_scoring and args.experimental_c_newborns
    assert args.native_threads == 2
    assert args.native_min_rows == 1
