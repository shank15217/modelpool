"""Pool CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelPool Proxy")
    parser.add_argument("--config", "-c", default="pool.yaml")
    parser.add_argument("--registry", "-r", default="resources.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=9000)
    args = parser.parse_args()
    print("ModelPool Pool - not yet implemented")
    print(f"  Registry: {args.registry}")
    print(f"  Config: {args.config}")
    print(f"  Listen: {args.host}:{args.port}")
    sys.exit(0)


if __name__ == "__main__":
    main()
