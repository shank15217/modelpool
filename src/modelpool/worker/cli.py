"""Worker CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from modelpool.registry import Registry
from modelpool.worker.loader import LlamaServerManager
from modelpool.worker.watchdog import Watchdog
from modelpool.worker.server import app, configure


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelPool Worker Agent")
    parser.add_argument(
        "--config", "-c",
        default="worker.yaml",
        help="Worker config file (default: worker.yaml)",
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
        default=9100,
        help="Listen port (default: 9100)",
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

    # Load worker config (simple YAML for now)
    import yaml
    from pathlib import Path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Worker config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        wconfig = yaml.safe_load(f)

    worker_name = wconfig.get("worker_id", "unknown")
    inference_port = wconfig.get("inference_port", 8080)
    log_dir = wconfig.get("log_dir", "/var/log/modelpool")
    idle_shutdown = wconfig.get("idle_shutdown", 0)

    # Get worker from registry for settings
    try:
        worker = registry.get_worker(worker_name)
        inference_port = worker.inference_port
    except Exception:
        pass  # Use config file values

    # Create manager, watchdog, configure app
    manager = LlamaServerManager(inference_port=inference_port, log_dir=log_dir)
    watchdog = Watchdog(manager, registry, worker_name)
    configure(manager, registry, watchdog, worker_name, idle_shutdown=idle_shutdown)

    # Start idle -- don't load any model until the pool requests one
    if idle_shutdown > 0:
        logging.info(f"Idle shutdown enabled ({idle_shutdown}s). Starting idle, no model loaded.")
    else:
        # Legacy behavior: load default resource on startup
        try:
            default_resource = registry.get_default_resource(worker_name)
            logging.info(f"Loading default resource: {default_resource.name}")
            manager.start(default_resource)
        except Exception as e:
            logging.warning(f"Could not load default resource: {e}")
            logging.warning("Worker starting in idle state")

    # Start serving
    logging.info(f"Worker '{worker_name}' listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
