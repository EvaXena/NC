#!/bin/bash
set -e

echo "============================================"
echo "  NAUTILUS Coral Detection - Merge & Eval"
echo "============================================"

PROJECT_ROOT="/root/nautilus/NAUTILUS"
QWEN_DIR="${PROJECT_ROOT}/qwen-vl-finetune"

# Paths
CHECKPOINT="/root/autodl-tmp/nautilus_checkpoints/checkpoint-1500"
MERGED_MODEL="/dev/shm/nautilus_coral_det_merged"
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DINO_PATH="/root/autodl-tmp/nautilus_checkpoints/weight/dino_vitl.pth"

TEST_JSON="${PROJECT_ROOT}/dataset/test_nautilus.json"
IMAGE_DIR="${PROJECT_ROOT}/dataset/test"
RESULT_JSON="/root/autodl-tmp/nautilus_eval_results.json"

# ========================================
# Step 1: Merge LoRA
# ========================================
echo ""
echo "[Step 1/2] Merging LoRA weights..."
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output:     ${MERGED_MODEL}"
echo "----------------------------------------"

if [ -d "${MERGED_MODEL}" ] && [ -f "${MERGED_MODEL}/config.json" ]; then
    echo "Merged model already exists, skipping merge."
else
    cd ${QWEN_DIR}
    python scripts/merge_lora.py \
        --input_path "${CHECKPOINT}" \
        --output_path "${MERGED_MODEL}" \
        --base_model "${BASE_MODEL}" \
        --dino_path "${DINO_PATH}"
    echo "[Step 1] Done"
fi

# ========================================
# Step 2: Batch Inference
# ========================================
echo ""
echo "[Step 2/2] Running batch inference on test set..."
echo "  Model:   ${MERGED_MODEL}"
echo "  Test:    ${TEST_JSON} (499 images)"
echo "  Output:  ${RESULT_JSON}"
echo "----------------------------------------"

cd ${QWEN_DIR}
python scripts/batch_inference.py \
    --checkpoint "${MERGED_MODEL}" \
    --test_json "${TEST_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --output "${RESULT_JSON}"

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Results: ${RESULT_JSON}"
echo "============================================"
