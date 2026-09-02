#!/usr/bin/env python3
"""
run_experiment_batch.py — Run run_experiments.py over a list of YAML suite configs.

Each YAML is run as a separate subprocess.  If one fails (non-zero exit code or
exception), the failure is recorded and the script continues with the remaining
configs.  Failed configs are NOT retried.

Usage
-----
    python run_experiment_batch.py config/experiments/suite_a.yaml \\
                                   config/experiments/suite_b.yaml \\
                                   config/experiments/suite_c.yaml

    # Or pass a text file containing one YAML path per line:
    python run_experiment_batch.py --list my_batch.txt

    # Forward extra flags to every run_experiments.py invocation:
    python run_experiment_batch.py suites/*.yaml -- --only gnn_baseline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run run_experiments.py over multiple YAML suite configs in sequence.')
    parser.add_argument(
        'configs', nargs='*', metavar='CONFIG.yaml',
        help='One or more experiment suite YAML files to run in order.')
    parser.add_argument(
        '--list', metavar='FILE',
        help='Text file containing one YAML path per line (blank lines and '
             'lines starting with # are ignored).')
    parser.add_argument(
        '--', dest='extra_args', nargs=argparse.REMAINDER, default=[],
        help='Extra arguments forwarded verbatim to every run_experiments.py call '
             '(e.g. -- --dry-run  or  -- --only gnn_baseline).')
    return parser.parse_args()


def _load_list_file(path: str) -> list[str]:
    lines = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith('#'):
                lines.append(line)
    return lines


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f'{h}h {m}m {s}s'
    if m:
        return f'{m}m {s}s'
    return f'{s}s'


def main():
    args = _parse_args()

    # Collect YAML paths from positional args and/or --list file
    configs = list(args.configs)
    if args.list:
        configs += _load_list_file(args.list)

    # Strip a leading '--' that argparse may leave in extra_args
    extra = [a for a in (args.extra_args or []) if a != '--']

    if not configs:
        print('ERROR: no config files specified.', file=sys.stderr)
        print('  Usage: python run_experiment_batch.py suite_a.yaml suite_b.yaml',
              file=sys.stderr)
        sys.exit(1)

    # Validate paths up front so we fail fast on typos
    missing = [c for c in configs if not Path(c).exists()]
    if missing:
        print('ERROR: the following config files do not exist:', file=sys.stderr)
        for m in missing:
            print(f'  {m}', file=sys.stderr)
        sys.exit(1)

    sep = '=' * 70
    print(f'\n{sep}')
    print(f'Batch runner: {len(configs)} suite(s) queued')
    if extra:
        print(f'  Extra args forwarded: {extra}')
    for i, c in enumerate(configs, 1):
        print(f'  {i:2d}. {c}')
    print(sep)

    results = []   # list of (config, status, duration_s, returncode)
    t_batch_start = time.time()

    for idx, config in enumerate(configs, 1):
        print(f'\n{sep}')
        print(f'[{idx}/{len(configs)}]  Starting: {config}')
        print(sep)

        cmd = [sys.executable, 'run_experiments.py', config] + extra
        t_start = time.time()
        status = 'ok'
        returncode = 0

        try:
            proc = subprocess.run(cmd)
            returncode = proc.returncode
            if returncode != 0:
                status = 'failed'
                print(f'\n  !! run_experiments.py exited with code {returncode} '
                      f'for: {config}', file=sys.stderr)
        except Exception as exc:
            returncode = -1
            status = 'error'
            print(f'\n  !! Exception while running {config}: {exc}', file=sys.stderr)

        duration = time.time() - t_start
        results.append((config, status, duration, returncode))
        print(f'\n  [{idx}/{len(configs)}]  {status.upper()}  '
              f'({_fmt_duration(duration)})  {config}')

    # ── final summary ─────────────────────────────────────────────────────────
    total_duration = time.time() - t_batch_start
    n_ok     = sum(1 for _, s, _, _ in results if s == 'ok')
    n_failed = len(results) - n_ok

    print(f'\n{sep}')
    print(f'Batch complete: {n_ok}/{len(configs)} succeeded  '
          f'({_fmt_duration(total_duration)} total)')
    print(sep)
    col = f"  {'Status':<8} {'Duration':>10}  Config"
    print(col)
    print('  ' + '-' * (len(col) - 2))
    for config, status, duration, rc in results:
        tag = 'OK' if status == 'ok' else f'FAIL({rc})'
        print(f'  {tag:<8} {_fmt_duration(duration):>10}  {config}')
    print()

    if n_failed:
        print(f'Failed configs ({n_failed}):')
        for config, status, _, rc in results:
            if status != 'ok':
                print(f'  exit={rc}  {config}')
        sys.exit(1)


if __name__ == '__main__':
    main()
