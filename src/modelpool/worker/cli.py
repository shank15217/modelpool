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
    pool_secret = wconfig.get("pool_secret")
    resource_name = wconfig.get("resource")  # model to load at boot

    # Get worker from registry for settings (including pool_secret)
    try:
        worker = registry.get_worker(worker_name)
        inference_port = worker.inference_port
        # Registry secret takes precedence if set
        if worker.pool_secret:
            pool_secret = worker.pool_secret
    except Exception:
        pass  # Use config file values

    # Create manager, watchdog, configure app
    manager = LlamaServerManager(inference_port=inference_port, log_dir=log_dir)
    watchdog = Watchdog(manager)
    configure(manager, watchdog, pool_secret=pool_secret)

    # Load the assigned model at boot (static pool: one model, served forever)
    if resource_name:
        try:
            resource = registry.get_resource(resource_name)
            logging.info(f"Loading resource: {resource.name}")
            manager.start(resource)
        except Exception as e:
            logging.error(f"Failed to load resource '{resource_name}': {e}")
            logging.warning("Worker starting in idle state")
    else:
        # Try to find the resource from the worker's assigned resources
        try:
            resources = registry.get_resources_for_worker(worker_name)
            if resources:
                resource = resources[0]
                logging.info(f"Loading first assigned resource: {resource.name}")
                manager.start(resource)
            else:
                logging.warning("No resource specified and no resources assigned to worker")
        except Exception as e:
            logging.warning(f"Could not determine resource to load: {e}")

    # Start serving
    logging.info(f"Worker '{worker_name}' listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
