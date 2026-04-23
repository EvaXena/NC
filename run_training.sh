#!/bin/bash
set -e

echo "============================================"
echo "  NAUTILUS Coral Detection LoRA Fine-tuning"
echo "  完整启动流程"
echo "============================================"

PROJECT_ROOT="/root/nautilus/NAUTILUS"
QWEN_DIR="${PROJECT_ROOT}/qwen-vl-finetune"
WEIGHT_DIR="${QWEN_DIR}/weight"
DINO_SRC_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"

# ========================================
# Step 1: 创建 conda 环境 & 安装依赖
# ========================================
echo ""
echo "[Step 1/5] 环境与依赖安装"
echo "----------------------------------------"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda 未找到，请先安装 Anaconda/Miniconda"
    exit 1
fi

# 创建环境（如果不存在）
if ! conda env list | grep -q "nautilus_qwen"; then
    echo "创建 conda 环境: nautilus_qwen (Python 3.10)..."
    conda create -n nautilus_qwen python=3.10 -y
else
    echo "conda 环境 nautilus_qwen 已存在，跳过创建"
fi

echo "激活环境并安装依赖..."
eval "$(conda shell.bash hook)"
conda activate nautilus_qwen

echo "安装 requirements.txt..."
pip install -r ${QWEN_DIR}/requirements.txt

echo "安装 flash-attn（编译较慢，约 10-30 分钟）..."
pip install flash-attn==2.7.3 --no-build-isolation || {
    echo "flash-attn 编译安装失败，尝试下载预编译 wheel..."
    echo "请手动从 https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3 下载对应 wheel"
    exit 1
}

echo "[Step 1] 完成 ✓"

# ========================================
# Step 2: 下载 Qwen2.5-VL-7B-Instruct 模型
# ========================================
echo ""
echo "[Step 2/5] 下载 Qwen2.5-VL-7B-Instruct 基座模型"
echo "----------------------------------------"

# 如果本地已有模型路径，修改下面的变量指向本地路径即可跳过下载
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

python3 -c "
from huggingface_hub import snapshot_download
print('开始下载 Qwen2.5-VL-7B-Instruct...')
path = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')
print(f'模型已下载到: {path}')
" || {
    echo "自动下载失败，请手动下载模型:"
    echo "  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct"
    echo "或使用 modelscope:"
    echo "  modelscope download Qwen/Qwen2.5-VL-7B-Instruct"
    echo ""
    echo "下载完成后修改训练脚本中的 pretrain_llm 变量指向本地路径"
    exit 1
}

echo "[Step 2] 完成 ✓"

# ========================================
# Step 3: 下载并处理 DINOv2 (DepthAnythingV2) 权重
# ========================================
echo ""
echo "[Step 3/5] 准备 DINOv2 权重"
echo "----------------------------------------"

mkdir -p ${WEIGHT_DIR}

if [ ! -f "${WEIGHT_DIR}/dino_vitl.pth" ]; then
    DAV2_PATH="${WEIGHT_DIR}/depth_anything_v2_vitl.pth"

    if [ ! -f "${DAV2_PATH}" ]; then
        echo "下载 Depth-Anything-V2-ViT-L 权重..."
        wget -O "${DAV2_PATH}" "${DINO_SRC_URL}" || {
            echo "自动下载失败，请手动下载:"
            echo "  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
            echo "放置到: ${DAV2_PATH}"
            exit 1
        }
    fi

    echo "提取 DINOv2 权重..."
    python3 ${PROJECT_ROOT}/utils/process_vitl_weight.py \
        --dav2-vitl "${DAV2_PATH}" \
        --dinov2-vitl "${WEIGHT_DIR}/dino_vitl.pth"

    echo "DINOv2 权重已生成: ${WEIGHT_DIR}/dino_vitl.pth"
else
    echo "dino_vitl.pth 已存在，跳过"
fi

echo "[Step 3] 完成 ✓"

# ========================================
# Step 4: 数据集格式转换
# ========================================
echo ""
echo "[Step 4/5] 数据集格式转换"
echo "----------------------------------------"

DATASET_DIR="${PROJECT_ROOT}/dataset"

if [ ! -f "${DATASET_DIR}/train_nautilus.json" ]; then
    echo "转换数据集格式..."
    python3 ${DATASET_DIR}/convert_to_nautilus_format.py
else
    echo "train_nautilus.json 已存在，跳过转换"
fi

# 验证
python3 -c "
import json
d = json.load(open('${DATASET_DIR}/train_nautilus.json'))
assert isinstance(d[0]['id'], list) and d[0]['id'][1] == '5', 'id 格式不正确'
print(f'数据集验证通过: {len(d)} 条训练样本')
"

echo "[Step 4] 完成 ✓"

# ========================================
# Step 5: 启动 LoRA 微调训练
# ========================================
echo ""
echo "[Step 5/5] 启动 LoRA 微调训练"
echo "----------------------------------------"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "训练配置:"
echo "  模型:    Qwen2.5-VL-7B-Instruct"
echo "  数据集:  coral_det_train (${DATASET_DIR}/train_nautilus.json)"
echo "  LoRA:    rank=128, alpha=256"
echo "  Epochs:  3"
echo "  BS:      4 (per device)"
echo "  LR:      2e-5 (LLM), 2e-7 (NAUTILUS modules)"
echo ""

cd ${QWEN_DIR}
bash scripts/nautilus_finetune/coral_det_lora.sh

echo ""
echo "============================================"
echo "  训练完成！"
echo "  输出目录: ${QWEN_DIR}/output/nautilus_coral_det_lora"
echo "============================================"
