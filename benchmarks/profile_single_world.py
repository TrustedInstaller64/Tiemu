"""Remote-only diagnostic driver for one world's remaining CPU bottlenecks.

The cases run sequentially and produce profiling evidence, not product-level
multi-world scheduling. Explicit boundary timers must preserve the checked
state/RNG digest. Optional cProfile data is marked incomplete because pybind11
bound methods can lose caller frames on the current Python runtime.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys


def text_file(path, limit=16384):
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')[:limit]
    except OSError:
        return None


def environment_snapshot(probe_perf):
    cpuinfo = text_file('/proc/cpuinfo', 262144)
    cpu_models, mhz = set(), []
    hypervisor_flag = False
    if cpuinfo:
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(':')
            if not separator:
                continue
            if key.strip() == 'model name':
                cpu_models.add(value.strip())
            elif key.strip() == 'cpu MHz':
                mhz.append(value.strip())
            elif key.strip() == 'flags':
                hypervisor_flag |= 'hypervisor' in value.split()
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    result = {
        'platform': platform.platform(), 'machine': platform.machine(),
        'logical_cpus': os.cpu_count(), 'allowed_cpu_affinity': affinity,
        'cpu_models': sorted(cpu_models), 'cpu_mhz_snapshot': mhz,
        'hypervisor_cpu_flag': hypervisor_flag if cpuinfo else None,
        'proc_self_cgroup': text_file('/proc/self/cgroup'),
        'cgroup_root_cpu_max': text_file('/sys/fs/cgroup/cpu.max'),
        'cgroup_root_cpu_stat': text_file('/sys/fs/cgroup/cpu.stat'),
        'cgroup_root_cpuset_effective': text_file('/sys/fs/cgroup/cpuset.cpus.effective'),
        'perf_event_paranoid': text_file('/proc/sys/kernel/perf_event_paranoid'),
        'scope': 'Metadata snapshot outside measured stages. Exposed CPU model/MHz need not '
                 'describe a dedicated host or sustained workload frequency. Cgroup root data '
                 'is reported with proc_self_cgroup and is not assumed to be the effective quota. '
                 'These fields do not establish IPC, memory bandwidth, cache misses or NUMA locality.'
    }
    perf = shutil.which('perf')
    probe = {'requested': probe_perf, 'executable': perf,
             'scope': 'User-space event availability probe on the true utility only; not a project measurement. '
                      'No hardware counts from this probe may be attributed to a simulation phase.'}
    if probe_perf and perf and platform.system() == 'Linux':
        command = [perf, 'stat', '-x,', '-e', 'cycles:u,instructions:u,cache-misses:u', '--', 'true']
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            probe.update(returncode=completed.returncode,
                         stdout=completed.stdout[:8192], stderr=completed.stderr[:8192],
                         command=command)
        except (OSError, subprocess.TimeoutExpired) as error:
            probe['error'] = str(error)
    elif probe_perf:
        probe['unavailable_reason'] = 'perf not installed or non-Linux host'
    result['perf_probe'] = probe
    return result


def compact_row(row):
    return {key: row[key] for key in
            ('file', 'line', 'function', 'calls', 'self_seconds', 'cumulative_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--probe-perf', action='store_true')
    parser.add_argument('--include-cprofile', action='store_true',
                        help='Retain known-incomplete cProfile evidence; never use its percentages.')
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('benchmark_single_world.py').resolve()
    environment = environment_snapshot(args.probe_perf)
    (output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'environment': environment}), flush=True)
    cases = [('human', 1000, 6, False), ('human', 10000, 3, False),
             ('human', 1000, 6, True), ('aeonic', 200, 2, False)]
    report = {
        'source_root': str(root), 'environment': environment, 'cases': [],
        'known_cprofile_issue': 'https://github.com/pybind/pybind11/issues/5600',
        'cprofile_rejected_for_percentages': True,
        'scope': 'Sequential isolated processes on one checkout. Each configuration has three '
                 'uninstrumented samples and one explicit boundary-timer sample, after separate warmup worlds. '
                 'An optional, separate cProfile sample is known incomplete and rejected for percentages. '
                 'First worker-pool startup is inside each measured stage if reached. '
                 'All samples share seed 731 and PYTHONHASHSEED=0. Profiler overhead is reported, '
                 'not subtracted. Inclusive boundary spans overlap; exclusive instrumented time '
                 'subtracts only instrumented descendants, not every function. Resource deltas '
                 'cover the measured stage and process '
                 'threads. Four-worker scoring is forced with threshold 1, but aeonic does not '
                 'use that human scoring pool. These short cases are hotspot diagnostics, not '
                 'stable speedup claims, a full-world run or a local-M1 measurement. '
                 'Digest covers entity order/serialized state/derived activity/stability/heroic '
                 'tendency/raw path-cache state, reincarnator identity by population index, '
                 'blueprint/seats, all three networks and optimizer states, and Python/NumPy/Torch '
                 'RNG. It does not cover every internal manager field or array alias relation.'
    }
    for phase, population, generations, reincarnator in cases:
        name = f'{phase}-{population}' + ('-reincarnator' if reincarnator else '')
        case = {'name': name, 'phase': phase, 'initial_population': population,
                'generations': generations, 'initial_reincarnator': reincarnator,
                'configurations': []}
        checked_digests = set()
        for workers in (1, 4):
            records = {}
            for mode in (('plain', 'boundary', 'profile') if args.include_cprofile else ('plain', 'boundary')):
                label = f'{name}-w{workers}-{mode}'
                result_path = output/f'{label}.json'
                command = [sys.executable, str(script), '--source-root', str(root),
                           '--phase', phase, '--population', str(population),
                           '--generations', str(generations), '--native-threads', str(workers),
                           '--native-min-rows', '1', '--repeats', '3' if mode == 'plain' else '1',
                           '--output', str(result_path)]
                if reincarnator:
                    command.append('--with-reincarnator')
                if mode == 'profile':
                    command.extend(['--profile-output-dir', str(output/f'{label}-stats')])
                if mode == 'boundary':
                    command.append('--boundary-timers')
                subprocess.run(command, check=True, timeout=180, stdout=subprocess.DEVNULL,
                               env={**os.environ, 'PYTHONHASHSEED': '0'})
                records[mode] = json.loads(result_path.read_text(encoding='utf-8'))
                for run in records[mode]['runs']:
                    checked_digests.add(run['state_digest'])
            plain, observed = records['plain'], records['boundary']
            boundary_run = observed['runs'][0]
            boundaries = boundary_run['boundaries']
            if not boundaries['valid']:
                raise SystemExit(f'{name}: explicit boundary integrity checks failed')
            item = {
                'workers_requested': workers, 'native_min_rows': 1,
                'plain_file': f'{name}-w{workers}-plain.json',
                'boundary_file': f'{name}-w{workers}-boundary.json',
                'plain_median_seconds': plain['median_seconds'],
                'boundary_seconds': boundary_run['seconds'],
                'boundary_to_plain_wall_ratio': boundary_run['seconds']/plain['median_seconds'],
                'plain_median_process_cpu_over_wall': statistics.median(
                    run['cpu_seconds']/run['seconds'] for run in plain['runs']),
                'boundary_process_cpu_over_wall': boundary_run['cpu_seconds']/boundary_run['seconds'],
                'plain_resource_deltas': [run['resource_delta'] for run in plain['runs']],
                'optimizer_steps': boundary_run['optimizer_steps'],
                'guide_optimizer_steps': boundary_run['guide_optimizer_steps'],
                'boundary_integrity_valid': boundaries['valid'],
                'boundary_spans': boundaries['spans'],
                'boundary_scope': boundaries['scope'],
            }
            if 'profile' in records:
                profile_run = records['profile']['runs'][0]
                profile = profile_run['profile']
                item['rejected_cprofile'] = {
                    'known_incomplete': True, 'rejected_for_percentages': True,
                    'seconds': profile_run['seconds'],
                    'wall_ratio_over_plain': profile_run['seconds']/plain['median_seconds'],
                    'total_self_seconds': profile['total_self_seconds'],
                    'total_calls': profile['total_calls'],
                    'integrity_violations': profile['integrity_violations'],
                    'hotspots': [compact_row(row) for row in profile['hotspots']],
                }
            case['configurations'].append(item)
            print(json.dumps({'case': name, **item}), flush=True)
        case['same_checked_state_and_rng_digest'] = len(checked_digests) == 1
        case['checked_digests'] = sorted(checked_digests)
        report['cases'].append(case)
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        if not case['same_checked_state_and_rng_digest']:
            raise SystemExit(f'{name}: profiling or worker count changed checked state/RNG; inspect evidence')
        print(json.dumps({'case': name, 'same_checked_state_and_rng_digest': True}), flush=True)


if __name__ == '__main__':
    main()
