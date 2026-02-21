#!/bin/bash
# 测量推理时间的辅助脚本
# 在inference.sh中调用，记录推理时间

PREFIX=$1
TIMING_FILE="${PREFIX}/inference_timing.txt"

# 记录推理开始时间
echo "=== Inference Timing ===" > "$TIMING_FILE"
echo "Start time: $(date +%s)" >> "$TIMING_FILE"
echo "Start time (human readable): $(date)" >> "$TIMING_FILE"

# 记录步骤3（网络推理）的开始时间
echo "Step 3 (network inference) start: $(date +%s)" >> "$TIMING_FILE"

# 执行实际的推理命令（通过参数传递）
shift
"$@"

# 记录推理结束时间
echo "Step 3 (network inference) end: $(date +%s)" >> "$TIMING_FILE"
echo "End time: $(date +%s)" >> "$TIMING_FILE"
echo "End time (human readable): $(date)" >> "$TIMING_FILE"

