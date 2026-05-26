# ModelPool Architecture

## Core Concept

A **resource** is a fully configured model recipe -- the exact command to launch an inference server, tuned for specific hardware and a specific use case. Resources come in two flavors:

- **Managed resources** -- local llama-server processes the worker starts, monitors, and stops
- **External resources** -- cloud APIs or pre-existing endpoints the pool can route to

The pool routes by task type, resolves to a resource, and either manages the lifecycle (managed) or just proxies (external). Clients don't know the difference.

```
Resource = Name + Exact launch command (or external endpoint) + Hardware target + Use case tags + Benchmarks
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Client     │     │              ModelPool                       │
│   (Hermes)   │────▶│                                              │
│              │     │  Router ──▶ Resource Registry ──▶ Workers    │
└─────────────┘     │                                              │
                    │  ┌─────────────┐  ┌──────────────────────┐   │
                    │  │  Managed    │  │  External            │   │
                    │  │  (local GPU)│  │  (cloud API)         │   │
                    │  │  lifecycle  │  │  proxy only           │   │
                    │  └─────────────┘  └──────────────────────┘   │
                    └──────────────────────────────────────────────┘
                              │                    │
                    ┌─────────▼──────┐   ┌─────────▼──────────┐
                    │  hwrouter      │   │  xAI cloud         │
                    │  llama-server  │   │  api.x.ai/v1       │
                    │  :8080         │   │  (already running)  │
                    └────────────────┘   └────────────────────┘
```

## How a Managed Resource Works

### Step 1: Define the resource in `resources.yaml`

Take the exact command you tested manually and put it in the resource definition:

```yaml
resources:
  qwen36-35b-a3b_mtp_no-reasoning_multi-gpu:
    description: "35B MoE, MTP speculative, reasoning off, both GPUs"
    type: managed
    size_gb: 21
    ctx: 262144
    capabilities: [compression, summarize, chat]
    workers: [hwrouter]
    benchmark:
      prompt_eval_tps: 3420
      generation_tps: 8.2
      tested_at: 2026-05-25
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, "/AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf"]
        - [-c, "262144"]
        - [-ngl, "99"]
        - [-fa, "on"]
        - [--cache-type-k, q8_0]
        - [--cache-type-v, q8_0]
        - [--reasoning, "off"]
        - [--jinja]
        - [--chat-template-kwargs, '{"preserve_thinking":true}']
        - [--no-mmap]
        - [--kv-unified]
        - [--tensor-split, "0.5,0.5"]
        - [--device, "ROCm0,ROCm1"]
        - [--parallel, "1"]
        - [--cont-batching]
        - [--batch-size, "1024"]
        - [--ubatch-size, "512"]
        - [--port, "{inference_port}"]       # template: replaced with worker's port
        - [--host, "0.0.0.0"]
        - [--temp, "0.7"]
        - [--top-p, "0.95"]
        - [--top-k, "40"]
        - [--min-p, "0.05"]
        - [--presence-penalty, "0.0"]
        - [--spec-type, draft-mtp]
        - [--spec-draft-n-max, "3"]
        - [--spec-draft-p-min, "0.75"]
        - [--cache-ram, "16384"]
```

The command is the exact thing you ran and tested. Flags are stored as a list of `[flag, value]` pairs (or `[flag]` for boolean flags). The only templating is `{inference_port}` -- everything else is literal.

### Step 2: Worker receives a load request

```
POST http://hwrouter:9100/worker/load
{ "resource": "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }
```

### Step 3: Worker executes the lifecycle

```
Worker state machine during load:

  READY (running qwen36-27b)
    │
    ▼
  DRAINING
    │  GET http://localhost:8080/props -> check slots_processing == 0
    │  Wait up to 30s for in-flight requests to finish
    │  If timeout: force stop anyway
    ▼
  STOPPING
    │  Send SIGTERM to llama-server process
    │  Wait up to 10s for clean exit
    │  If still running: SIGKILL
    │  Verify port 8080 is released
    ▼
  LOADING
    │  Build command line from resource definition:
    │    /AITOOLCHAIN/llama.cpp/build/bin/llama-server \
    │      -m /AITOOLCHAIN/models/.../Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf \
    │      -c 262144 -ngl 99 -fa on ... --port 8080 --host 0.0.0.0 ...
    │
    │  Launch as subprocess (no shell, direct exec)
    │  Capture stdout/stderr to log file
    │
    │  Poll health: GET http://localhost:8080/health
    │  Every 2s, up to 120s startup timeout
    │
    │  Health check returns 200 -> model is loaded and serving
    ▼
  READY (running qwen36-35b-a3b)
    Report to pool: resource loaded, ready for requests
```

### Step 4: Pool proxies requests

```
Client -> Pool (port 9000) -> Worker (port 8080) -> llama-server
                                 ^
                                 routes OpenAI /v1/chat/completions
```

The pool proxies the raw HTTP request to the worker's inference port. llama-server serves it directly. Responses stream back through the pool to the client.

### Step 5: Shutdown / revert

When the idle timer expires or pool requests a revert:

```
  READY (running qwen36-35b-a3b)
    │
    ▼
  DRAINING -> STOPPING -> LOADING (default resource) -> READY
```

The worker goes through the same drain -> stop -> start cycle to load the default resource.

### Why subprocess, not systemd?

The worker IS a systemd service (`modelpool-worker.service`). It's a long-running Python process that manages llama-server as child processes:

```
systemd
  └── modelpool-worker (Python, always running)
        ├── HTTP API on :9100 (management)
        └── child process management
              ├── llama-server (port 8080)  ← managed subprocess
              └── (only one at a time)
```

Benefits of subprocess over restarting systemd services:
- **No privilege escalation** -- worker runs as a user, doesn't need systemd reload
- **Faster** -- no systemd overhead on start/stop
- **Direct control** -- SIGTERM/SIGKILL, stdout/stderr capture, exit code checking
- **Portable** -- works on any Linux, no systemd-specific APIs

The worker process itself is managed by systemd (auto-restart on crash). The llama-server processes are managed by the worker (start/stop on demand).

## External Resources

Cloud APIs and pre-existing endpoints. No lifecycle management -- the pool just proxies.

```yaml
resources:
  grok-4.3_general:
    description: "Grok 4.3 via xAI OAuth, 256K ctx, cloud"
    type: external
    endpoint: https://api.x.ai/v1
    auth:
      method: xai-oauth       # uses Hermes auth store
    model: grok-4.3           # model name to send in requests
    ctx: 256000
    capabilities: [chat, compression, agentic, vision]
    workers: [cloud-xai]
    benchmark:
      prompt_eval_tps: null   # fill after testing
      generation_tps: null

  glm-45-flash_general:
    description: "GLM-4.5-Flash free tier, 131K ctx"
    type: external
    endpoint: https://open.bigmodel.cn/api/paas/v4
    auth:
      method: api_key
      env_var: LM_API_KEY
    model: glm-4.5-flash
    ctx: 131072
    capabilities: [chat, compression, summarize, triage]
    workers: [cloud-zai-free]
    benchmark:
      prompt_eval_tps: null
      generation_tps: null
```

External resources don't need a `command` section. They need:
- `endpoint` -- base URL for the API
- `auth` -- how to authenticate (OAuth reference, env var name, or static key)
- `model` -- model name to pass in the request body

The pool handles auth injection when proxying to external resources -- it reads the auth config, gets the current credentials, and adds them to the proxied request.

## Workers

Workers are targets that can serve resources. Two types:

```yaml
workers:
  # Managed: the worker agent runs on this host and controls llama-server
  hwrouter:
    host: 192.168.35.185
    worker_port: 9100          # worker agent API
    inference_port: 8080       # llama-server port
    type: managed
    vram_gb: 32
    max_model_gb: 28
    swap_timeout: 120
    drain_timeout: 30
    default_resource: qwen36-27b_mtp_vision_multi-gpu

  aitooolchain:
    host: 192.168.35.17
    worker_port: 9100
    inference_port: 8081
    type: managed
    ram_gb: 48
    max_model_gb: 40
    swap_timeout: 60
    drain_timeout: 10
    default_resource: qwen36-35b-a3b_no-reasoning_cpu

  # External: no worker agent, just an API endpoint
  cloud-xai:
    type: external
    # No host/ports -- the resource defines the endpoint

  cloud-zai-free:
    type: external
```

Managed workers run the worker agent. External workers are virtual -- they exist only in the routing table so the pool can direct traffic to them.

## Routing

Maps task types to resources. The pool tries the primary resource first, falls back if unavailable.

```yaml
routing:
  compression:
    resource: qwen36-35b-a3b_mtp_no-reasoning_multi-gpu
    fallback_resource: glm-45-flash_general     # free cloud first
    fallback_resource_2: grok-4.3_general       # then xAI
    fallback_resource_3: qwen36-35b-a3b_no-reasoning_cpu  # last resort: local CPU
    timeout: 120
    idle_revert: 300
    swap_behavior: queue

  code-review:
    resource: ornstein-27b-saber_q5_multi-gpu
    fallback_resource: qwen36-27b_mtp_vision_multi-gpu
    timeout: 300
    idle_revert: 180
    swap_behavior: queue

  triage:
    resource: glm-45-flash_general              # free, fast, cloud
    fallback_resource: qwen36-27b_mtp_vision_multi-gpu
    timeout: 30
    swap_behavior: fallback

  chat:
    resource: qwen36-27b_mtp_vision_multi-gpu
    fallback_resource: none
    timeout: 120
    idle_revert: 0
    swap_behavior: fallback
```

Fallback chain for compression:
```
1. Local 35B-A3B on GPU (free, fastest at long context)
2. GLM-4.5-Flash cloud (free, fast, but 131K ctx limit)
3. Grok 4.3 cloud (OAuth, 256K ctx, subscription)
4. Local 35B-A3B on CPU (free, but 12 min for 100K tokens)
```

The pool tries each in order. If a managed resource needs a swap, it waits (queue behavior). If a cloud resource fails, it immediately tries the next fallback.

## Data Flow (Full Example)

```
1. Hermes sends: POST http://localhost:9000/v1/chat/completions
   Header: X-Task-Type: compression
   Body: { messages: [...], max_tokens: 4096 }

2. Pool looks up routing: compression -> qwen36-35b-a3b_mtp_no-reasoning_multi-gpu

3. Pool checks: which worker serves this resource? -> hwrouter

4. Pool checks hwrouter: GET http://hwrouter:9100/worker/status
   Response: { resource: "qwen36-27b_mtp_vision_multi-gpu", state: "ready" }

5. Wrong resource loaded. Pool triggers swap:
   POST http://hwrouter:9100/worker/load
   { "resource": "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }
   Response: 202 { status: "loading", estimated_s: 60 }

6. Worker lifecycle:
   - Drains active slots (30s max)
   - Stops llama-server (SIGTERM, 10s, then SIGKILL)
   - Builds command from resource flags
   - Starts subprocess: /AITOOLCHAIN/llama.cpp/build/bin/llama-server -m ... --port 8080 ...
   - Polls GET http://localhost:8080/health every 2s
   - Health returns 200 after ~45s
   - Reports: { state: "ready", resource: "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }

7. Pool proxies the original request:
   POST http://hwrouter:8080/v1/chat/completions
   Body: { messages: [...], max_tokens: 4096 }
   (streaming response passes through pool back to Hermes)

8. Pool starts idle timer: 300s
   No compression requests for 5 minutes -> revert to qwen36-27b default
```

## Worker Internals

### Subprocess Management

```python
class LlamaServerManager:
    """Manages a single llama-server subprocess."""

    def __init__(self, worker_config):
        self.config = worker_config
        self.process: subprocess.Popen | None = None
        self.state: str = "idle"      # idle, loading, ready, draining, stopping, error
        self.loaded_resource: str | None = None
        self.log_file: file

    def start(self, resource_def: dict) -> None:
        """Start llama-server with exact command from resource definition."""
        # Build command from resource
        cmd = [resource_def["command"]["binary"]]
        for flag in resource_def["command"]["flags"]:
            cmd.extend(flag)
        
        # Replace template variables
        cmd = [s.replace("{inference_port}", str(self.config["inference_port"])) 
               for s in cmd]

        # Launch as direct subprocess (no shell)
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # process group for clean kill
        )
        self.state = "loading"

        # Wait for health check
        if not self._wait_healthy(timeout=self.config.get("swap_timeout", 120)):
            raise RuntimeError("llama-server failed to start")

        self.state = "ready"
        self.loaded_resource = resource_def["name"]

    def stop(self, timeout: int = 10) -> None:
        """Stop the running llama-server."""
        if not self.process:
            return

        # SIGTERM to process group
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # SIGKILL to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=5)

        self.process = None
        self.state = "idle"
        self.loaded_resource = None

    def drain(self, timeout: int = 30) -> None:
        """Wait for all in-flight requests to complete."""
        self.state = "draining"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"http://localhost:{self.config['inference_port']}/props",
                    timeout=2,
                )
                if resp.json().get("slots_processing", 1) == 0:
                    return  # drained
            except requests.ConnectionError:
                return  # server down, nothing to drain
            time.sleep(1)

        # Timeout: force proceed anyway

    def _wait_healthy(self, timeout: int) -> bool:
        """Poll /health until 200 OK."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"http://localhost:{self.config['inference_port']}/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(2)
        return False

    def get_status(self) -> dict:
        """Current status for /worker/status endpoint."""
        status = {
            "state": self.state,
            "loaded_resource": self.loaded_resource,
            "pid": self.process.pid if self.process else None,
        }
        
        if self.state == "ready":
            try:
                resp = requests.get(
                    f"http://localhost:{self.config['inference_port']}/health",
                    timeout=2,
                )
                health = resp.json()
                status["slots_idle"] = health.get("slots_idle", 0)
                status["slots_processing"] = health.get("slots_processing", 0)
            except:
                status["state"] = "error"
        
        return status
```

### Watchdog

The worker runs a background thread that checks llama-server health every 15s:

```python
async def watchdog():
    while True:
        await asyncio.sleep(15)
        if manager.state == "ready":
            try:
                resp = requests.get(
                    f"http://localhost:{config['inference_port']}/health",
                    timeout=5,
                )
                if resp.status_code != 200:
                    manager.state = "error"
            except:
                manager.state = "error"
                # Auto-recover: stop broken process, load default
                manager.stop()
                default = registry.get_default_resource(worker_id)
                manager.start(default)
```

## Auth for External Resources

The pool needs to inject credentials when proxying to external resources. Auth methods:

| Method | How it works |
|---|---|
| `xai-oauth` | Read tokens from `~/.hermes/auth.json`, refresh if expired, inject as `Authorization: Bearer <token>` |
| `api_key` | Read key from env var, inject as `Authorization: Bearer <key>` |
| `none` | No auth (local endpoints) |

```python
def inject_auth(request_headers: dict, auth_config: dict) -> dict:
    if auth_config["method"] == "xai-oauth":
        creds = resolve_xai_oauth_runtime_credentials()
        request_headers["Authorization"] = f"Bearer {creds['api_key']}"
    elif auth_config["method"] == "api_key":
        key = os.environ.get(auth_config["env_var"], "")
        request_headers["Authorization"] = f"Bearer {key}"
    return request_headers
```

## Deployment

```
Hermes Host (192.168.35.x)
├── modelpool-pool (port 9000)     # systemd service
└── resources.yaml                  # shared config

Managed Workers
├── hwrouter (192.168.35.185)
│   ├── modelpool-worker (port 9100)  # systemd service
│   ├── llama-server (port 8080)      # managed subprocess
│   └── resources.yaml                 # copy of shared config
└── aitooolchain (192.168.35.17)
    ├── modelpool-worker (port 9100)
    ├── llama-server (port 8081)
    └── resources.yaml

External Workers (no agent needed)
├── cloud-xai (api.x.ai)
└── cloud-zai-free (open.bigmodel.cn)
```

`resources.yaml` is the same file everywhere. Each worker reads only its own section.

## Security

- Managed workers: firewall worker port (9100) to pool host only
- Inference ports (8080/8081): firewall to pool host only
- External resources: auth tokens never logged, refreshed automatically
- Pool: firewall to Hermes host only
- No auth between pool and workers (homelab, firewalled)

## Adding a New Resource

1. SSH to the target worker
2. Test the llama-server command manually -- get it working perfectly
3. Benchmark it: `modelpool bench` or manual timing
4. Add the exact working command as a new resource in `resources.yaml`
5. Add routing entry mapping a task type to the new resource
6. Deploy updated `resources.yaml` to pool + workers
7. Test: `curl -H "X-Task-Type: <task>" http://pool:9000/v1/chat/completions -d '...'`

No parameter generation. No interpolation (except `{inference_port}`). What you tested is what runs.

## Future Extensions

- **Hot standby:** Pre-load the next most likely resource on idle workers
- **Auto-benchmark:** On first load, run a quick benchmark, store results in registry
- **Multi-model:** Split GPUs to run two small resources simultaneously
- **Priority preemption:** High-priority tasks interrupt low-priority loads
- **Resource versioning:** Track command changes, re-benchmark on update
