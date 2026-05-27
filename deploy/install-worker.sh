#!/usr/bin/env bash
# Install ModelPool Worker on a remote inference host.
# Usage: ./install-worker.sh <worker-id> <host> [ssh-key] [ssh-user]
#
# Example:
#   ./install-worker.sh hwrouter 192.168.35.185 /root/.ssh/proxmox-vm-key root

set -euo pipefail

WORKER_ID="${1:?Usage: install-worker.sh <worker-id> <host> [ssh-key] [ssh-user]}"
HOST="${2:?Usage: install-worker.sh <worker-id> <host> [ssh-key] [ssh-user]}"
SSH_KEY="${3:-}"
SSH_USER="${4:-root}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

SSH="ssh $SSH_OPTS ${SSH_USER}@${HOST}"
SCP="scp $SSH_OPTS"

echo "=== Installing ModelPool Worker: $WORKER_ID on $HOST ==="

# 1. Copy source
echo "[1/5] Copying source to $HOST..."
$SCP -r /root/modelpool/src /root/modelpool/pyproject.toml ${SSH_USER}@${HOST}:/tmp/modelpool/ 2>/dev/null || {
    $SSH "mkdir -p /tmp/modelpool/src"
    $SCP -r /root/modelpool/src/* ${SSH_USER}@${HOST}:/tmp/modelpool/src/
    $SCP /root/modelpool/pyproject.toml ${SSH_USER}@${HOST}:/tmp/modelpool/
}

# 2. Copy config
echo "[2/5] Copying config..."
$SSH "mkdir -p /etc/modelpool /opt/modelpool /var/log/modelpool"
$SCP /root/modelpool/resources.yaml ${SSH_USER}@${HOST}:/etc/modelpool/resources.yaml

# 3. Generate worker.yaml for this host
echo "[3/5] Generating worker config..."
$SSH "cat > /etc/modelpool/worker.yaml << 'EOF'
worker_id: $WORKER_ID
log_dir: /var/log/modelpool
EOF"

# 4. Install Python package
echo "[4/5] Installing Python package..."
$SSH << 'REMOTE'
set -e
rm -rf /opt/modelpool/src /opt/modelpool/pyproject.tomv
cp -r /tmp/modelpool/src /opt/modelpool/
cp /tmp/modelpool/pyproject.toml /opt/modelpool/

# Create venv and install
if [ ! -d /opt/modelpool/venv ]; then
    python3 -m venv /opt/modelpool/venv 2>/dev/null || uv venv /opt/modelpool/venv --python 3.11
fi
/opt/modelpool/venv/bin/pip install -e "/opt/modelpool[dev]" 2>/dev/null || \
    /opt/modelpool/venv/bin/uv pip install -e "/opt/modelpool[dev]" 2>/dev/null || {
        # Fallback: install deps manually
        /opt/modelpool/venv/bin/pip install fastapi uvicorn httpx requests pyyaml || true
    }

echo "Package installed"
REMOTE

# 5. Install and start systemd service
echo "[5/5] Installing systemd service..."
$SCP /root/modelpool/deploy/modelpool-worker.service ${SSH_USER}@${HOST}:/etc/systemd/system/modelpool-worker.service
$SSH << REMOTE
set -e
systemctl daemon-reload
systemctl enable modelpool-worker.service

# Stop old llama-server if running (modelpool takes over port 8080)
if systemctl is-active llama-server.service 2>/dev/null; then
    echo "Stopping existing llama-server.service..."
    systemctl stop llama-server.service
    systemctl disable llama-server.service
fi

# Start modelpool worker
echo "Starting modelpool-worker..."
systemctl start modelpool-worker.service
sleep 3
systemctl status modelpool-worker.service --no-pager || true
echo ""
echo "Worker should be listening on port 9100 (management) and 8080 (inference)"
echo "Test: curl http://$HOST:9100/worker/status"
REMOTE

echo ""
echo "=== Done: $WORKER_ID on $HOST ==="
