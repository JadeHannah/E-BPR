config=$1
ckpt=$2
coarse_dir=$3
prefix=$4

IOU_THRESH=${IOU_THRESH:-0.55}  # 补丁过滤阈值（原论文使用0.55）
NMS_THRESH=${NMS_THRESH:-0.55}   # NMS阈值，用于补丁检测去重（固定为0.55）
USE_BIOU=${USE_BIOU:-0}  # 0 for IoU, 1 for BIoU
BIOU_DILATION_RATIO=${BIOU_DILATION_RATIO:-0.02}
IMG_DIR=${IMG_DIR:-'data/cityscapes/leftImg8bit/val'}
GT_JSON=${GT_JSON:-'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'}
BPR_ROOT=${BPR_ROOT:-'.'}
GPUS=${GPUS:-1}

coarse_json=${prefix}/coarse.json
out_pkl=${prefix}/refined.pkl
out_json=${prefix}/refined.json
out_dir=${prefix}/refined
dataset_dir=${prefix}/patches


set -x
GREEN='\033[0;32m'
END='\033[0m\n'

mkdir -p $prefix

# 检查是否已完成
if [ -f "$out_json" ]; then
    echo "Result already exists: $out_json"
    echo "Skipping inference. Delete the file to re-run."
    exit 0
fi

printf ${GREEN}"convert to json format ..."${END}
if [ ! -f "$coarse_json" ]; then
    python $BPR_ROOT/tools/cityscapes2json.py \
        $coarse_dir \
        $GT_JSON \
        $coarse_json
else
    echo "Coarse JSON already exists, skipping..."
fi

printf ${GREEN}"build patches dataset ..."${END}
echo ""
echo "=== Environment Variables ==="
echo "  IOU_THRESH: ${IOU_THRESH:-not set (using default 0.55)}"
echo "  USE_BIOU: ${USE_BIOU:-not set (using default 0)}"
echo "  BIOU_DILATION_RATIO: ${BIOU_DILATION_RATIO:-not set (using default 0.02)}"
echo "  IMG_DIR: ${IMG_DIR:-not set (using default)}"
echo "  GT_JSON: ${GT_JSON:-not set (using default)}"
echo "  BPR_ROOT: ${BPR_ROOT:-not set (using default .)}"
echo "  GPUS: ${GPUS:-not set (using default 1)}"
echo "=============================="
echo ""

# 自动检测数据集类型（val或test）
if [[ "$IMG_DIR" == *"test"* ]]; then
    DATASET_MODE="test"
else
    DATASET_MODE="val"
fi
echo "Detected dataset mode: $DATASET_MODE"

if [ ! -d "$dataset_dir/detail_dir/$DATASET_MODE" ] || [ -z "$(ls -A $dataset_dir/detail_dir/$DATASET_MODE 2>/dev/null)" ]; then
    echo "Running split_patches.py..."
    echo "  Coarse JSON: $coarse_json"
    echo "  GT JSON: $GT_JSON"
    echo "  Image dir: $IMG_DIR"
    echo "  Output dir: $dataset_dir"
    echo "  IOU_THRESH (filter): $IOU_THRESH (原论文使用0.55)"
    echo "  NMS_THRESH (detection): $NMS_THRESH (原论文使用0.25)"
    echo "  USE_BIOU: $USE_BIOU"
    echo "  BIOU_DILATION_RATIO: $BIOU_DILATION_RATIO"
    
    if [ "$USE_BIOU" -eq 1 ]; then
            python $BPR_ROOT/tools/split_patches.py \
                $coarse_json \
                $GT_JSON \
                $IMG_DIR \
                $dataset_dir \
                --iou-thresh $IOU_THRESH \
                --nms-thresh $NMS_THRESH \
                --use-biou \
                --biou-dilation-ratio $BIOU_DILATION_RATIO \
                --biou-small-patch-dilation 5 \
                --mode $DATASET_MODE
        SPLIT_EXIT_CODE=$?
    else
        python $BPR_ROOT/tools/split_patches.py \
            $coarse_json \
            $GT_JSON \
            $IMG_DIR \
            $dataset_dir \
            --iou-thresh $IOU_THRESH \
            --nms-thresh $NMS_THRESH \
            --mode $DATASET_MODE
        SPLIT_EXIT_CODE=$?
    fi
    
    if [ $SPLIT_EXIT_CODE -ne 0 ]; then
        echo "ERROR: split_patches.py failed with exit code $SPLIT_EXIT_CODE"
        exit 1
    fi
    
    # 检查目录是否被创建
    if [ ! -d "$dataset_dir/detail_dir/$DATASET_MODE" ]; then
        echo "WARNING: detail_dir/$DATASET_MODE was not created after split_patches.py"
        echo "  This might indicate all patches were filtered out"
        # 确保目录存在（即使为空）
        mkdir -p "$dataset_dir/detail_dir/$DATASET_MODE"
    fi
    
    echo "split_patches.py completed. Checking results..."
    DETAIL_COUNT=$(find "$dataset_dir/detail_dir/$DATASET_MODE" -name "*.txt" 2>/dev/null | wc -l)
    echo "  Detail files created: $DETAIL_COUNT"
else
    echo "Patches dataset already exists, skipping..."
fi

printf ${GREEN}"inference the network ..."${END}
if [ ! -f "$out_pkl" ]; then
    export DATA_ROOT=$dataset_dir
    echo "Setting DATA_ROOT=$DATA_ROOT"
    TIMING_FILE="${prefix}/inference_timing.txt"
    
    # 记录推理开始时间
    echo "=== Inference Timing ===" > "$TIMING_FILE"
    echo "Start time: $(date +%s)" >> "$TIMING_FILE"
    echo "Start time (human readable): $(date)" >> "$TIMING_FILE"
    START_TIME=$(date +%s)
    
    if [ "$GPUS" -eq 1 ]; then
        # Single GPU inference
        # Check CUDA availability first
        echo "Checking GPU availability..."
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "⚠️  WARNING: CUDA may not be available, model will run on CPU"
        echo ""
        
        python $BPR_ROOT/tools/test_float.py \
            $config \
            $ckpt \
            --out $out_pkl
    else
        # Multi GPU inference
        bash $BPR_ROOT/tools/dist_test_float.sh \
            $config \
            $ckpt \
            $GPUS \
            --out $out_pkl
    fi
    
    # 记录推理结束时间
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "End time: $END_TIME" >> "$TIMING_FILE"
    echo "End time (human readable): $(date)" >> "$TIMING_FILE"
    echo "Elapsed time: ${ELAPSED} seconds" >> "$TIMING_FILE"
    if command -v date &> /dev/null && date --version &> /dev/null 2>&1; then
        echo "Elapsed time (human readable): $(date -d @${ELAPSED} -u +%H:%M:%S 2>/dev/null || echo 'N/A')" >> "$TIMING_FILE"
    fi
else
    echo "Network output already exists, skipping inference..."
fi

printf ${GREEN}"reassemble ..."${END}
if [ ! -f "$out_json" ]; then
    python $BPR_ROOT/tools/merge_patches.py \
        $coarse_json \
        $GT_JSON \
        $out_pkl \
        $dataset_dir/detail_dir/$DATASET_MODE \
        $out_json
else
    echo "Refined JSON already exists, skipping merge..."
fi

printf ${GREEN}"convert to cityscape format ..."${END}
if [ ! -d "$out_dir" ] || [ -z "$(ls -A $out_dir 2>/dev/null)" ]; then
    python $BPR_ROOT/tools/json2cityscapes.py \
        $out_json \
        $GT_JSON \
        $out_dir
else
    echo "Cityscapes format output already exists, skipping..."
fi