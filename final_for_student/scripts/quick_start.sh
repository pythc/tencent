#!/bin/bash
# 快速开始脚本 - 一键准备数据和开始训练

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "StarRocks NL→SQL 训练流程"
echo "=========================================="

# 激活虚拟环境
VENV_PATH="/home/ubuntu/workspace/tencent/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
fi

cd "$PROJECT_DIR"

# 1. 准备 CSpider 数据
echo ""
echo "Step 1: Preparing CSpider data..."
python scripts/prepare_cspider.py \
    --train_json /home/ubuntu/workspace/tencent/full_CSpider/CSpider/train.json \
    --gold_sql /home/ubuntu/workspace/tencent/full_CSpider/CSpider/train_gold.sql \
    --tables_json /home/ubuntu/workspace/tencent/full_CSpider/CSpider/tables.json \
    --output data/cspider_train.json || {
    echo "Error: Failed to prepare CSpider data"
    exit 1
}

# 2. 准备域内数据
echo ""
echo "Step 2: Preparing domain data..."
python scripts/prepare_domain_data.py \
    --input data/sft_conversation.json \
    --output_sft data/domain_train.json \
    --output_orpo data/domain_orpo.json || {
    echo "Error: Failed to prepare domain data"
    exit 1
}

# 3. 注册数据集
echo ""
echo "Step 3: Registering datasets..."
python scripts/register_datasets.py || {
    echo "Error: Failed to register datasets"
    exit 1
}

echo ""
echo "=========================================="
echo "Data preparation completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Train Stage A (General):"
echo "   cd /home/ubuntu/workspace/LLaMA-Factory"
echo "   llamafactory-cli train $PROJECT_DIR/configs/stage_a_sft.yaml"
echo ""
echo "2. Train Stage B (Domain):"
echo "   llamafactory-cli train $PROJECT_DIR/configs/stage_b_sft.yaml"
echo ""
echo "3. Train Stage B ORPO (Optional):"
echo "   llamafactory-cli train $PROJECT_DIR/configs/stage_b_orpo.yaml"
echo ""
echo "4. Evaluate:"
echo "   python $PROJECT_DIR/scripts/evaluate.py \\"
echo "       --model_path saves/stage_b_sft \\"
echo "       --samples_file $PROJECT_DIR/data/success_samples.jsonl"
echo ""

