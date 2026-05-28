"""Benchmark CLI entry point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelPool Benchmark")
    parser.add_argument("resource", nargs="?", help="Resource name to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all managed resources")
    parser.add_argument("--parallel", action="store_true", help="Run parallel slot benchmark")
    parser.add_argument("--endpoint", "-e", help="Override endpoint URL")
    parser.add_argument("--max-ctx", type=int, default=32000, help="Max context size to test")
    parser.add_argument("--max-slots", type=int, default=3, help="Max parallel slots to test")
    parser.add_argument("--update", action="store_true", help="Update resources.yaml with results")
    args = parser.parse_args()

    if not args.resource and not args.all and not args.endpoint:
        parser.print_help()
        print("\nExamples:")
        print("  modelpool-bench --endpoint http://192.168.35.185:8080")
        print("  modelpool-bench --endpoint http://192.168.35.185:8080 --parallel --max-slots 2")
        print("  modelpool-bench qwen36-27b_mtp_reasoning_multi-gpu")
        print("  modelpool-bench --all")
        sys.exit(0)

    if args.parallel:
        script = os.path.join(BENCH_DIR, "bench_parallel.py")
    else:
        script = os.path.join(BENCH_DIR, "bench_resource.py")

    cmd = [sys.executable, script]
    if args.endpoint:
        cmd += ["--endpoint", args.endpoint]
    if args.resource:
        cmd += ["--resource", args.resource]
    if args.max_ctx:
        cmd += ["--max-ctx", str(args.max_ctx)]
    if args.max_slots:
        cmd += ["--max-slots", str(args.max_slots)]

    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
