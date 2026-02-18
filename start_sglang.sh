#!/bin/bash
# AgentCPM-Explore SGLang Server Start Script
#
# Prerequisites:
#   pip install sglang[all]
#   Download model: https://huggingface.co/openbmb/AgentCPM-Explore
#
# Usage:
#   ./start_sglang.sh
#   MODEL_PATH=/path/to/model ./start_sglang.sh
#   GPU_ID=1 ./start_sglang.sh

# Configuration - modify these as needed
# Download: https://huggingface.co/openbmb/AgentCPM-Explore (~8GB)
MODEL_PATH="${MODEL_PATH:-$HOME/models/AgentCPM-Explore}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-30001}"
LOG_FILE="/tmp/sglang_agentcpm.log"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    echo ""
    echo "Please set MODEL_PATH environment variable or edit this script."
    echo "Example: MODEL_PATH=/home/user/models/AgentCPM-Explore ./start_sglang.sh"
    exit 1
fi

echo "Starting AgentCPM-Explore on port $PORT (GPU $GPU_ID)..."
echo "Model: $MODEL_PATH"
echo "Log: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
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
