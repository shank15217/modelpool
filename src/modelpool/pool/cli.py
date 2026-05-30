"""Pool CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from modelpool.registry import Registry
from modelpool.pool.router import Router
from modelpool.pool.proxy import PoolProxy
from modelpool.pool.server import app, configure


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelPool Proxy")
    parser.add_argument(
        "--config", "-c",
        default="pool.yaml",
        help="Pool config file (default: pool.yaml)",
    )
    parser.add_argument(
        "--registry", "-r",
        default="resources.yaml",
        help="Resource registry file (default: resources.yaml)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Listen host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9000,
        help="Listen port (default: 9000)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load registry
    try:
        registry = Registry.from_file(args.registry)
    except Exception as e:
        print(f"Failed to load registry: {e}", file=sys.stderr)
        sys.exit(1)

    # Create router and proxy
    router = Router(registry)
    proxy = PoolProxy(registry, router)
    configure(registry, router, proxy)

    # Start serving
    logging.info(f"ModelPool proxy listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
