"""Benchmark CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelPool Benchmark")
    parser.add_argument("resource", nargs="?", help="Resource to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all resources")
    parser.add_argument("--update", action="store_true", help="Update resources.yaml with results")
    args = parser.parse_args()
    print("ModelPool Benchmark - not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
