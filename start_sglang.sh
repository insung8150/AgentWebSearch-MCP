#!/bin/bash
# AgentCPM-Explore SGLang Server Start Script
# Runs on RTX 3080 Ti (GPU 1) - workaround for Blackwell compatibility issues

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/sglang_agentcpm.log"
PORT=30001

cd "$SCRIPT_DIR"
source venv/bin/activate

echo "Starting AgentCPM-Explore on port $PORT (GPU 1: RTX 3080 Ti)..."
echo "Log: $LOG_FILE"

# Run on GPU 1 (RTX 3080 Ti, SM 8.6) with triton backend
# Blackwell GPU (SM 12.0) causes flashinfer JIT failure as nvcc doesn't support it
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path /mnt/nvme_4raid/model/AgentCPM-Explore \
    --port $PORT \
    --host 0.0.0.0 \
    --attention-backend triton \
    --sampling-backend pytorch \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo ""
echo "Wait 30-45 seconds for model loading..."
echo "Check status: curl http://localhost:$PORT/health"
echo "View logs: tail -f $LOG_FILE"
