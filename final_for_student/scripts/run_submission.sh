#!/bin/bash
# 生成提交文件的便捷脚本
# 使用方法: bash scripts/run_submission.sh [checkpoint_number]

set -e

# 激活虚拟环境
source /home/ubuntu/workspace/tencent/.venv/bin/activate

# 设置路径
BASE_MODEL_PATH="/home/ubuntu/workspace/tencent/models/Qwen_Qwen2.5-14B-Instruct"
PROJECT_DIR="/home/ubuntu/workspace/tencent/final_for_student"
LLAMA_FACTORY_DIR="/home/ubuntu/workspace/LLaMA-Factory"

# 获取checkpoint编号（默认使用最新的）
if [ -z "$1" ]; then
    # 自动查找最新的checkpoint
    LATEST_CHECKPOINT=$(find "$LLAMA_FACTORY_DIR/saves/stage_b_orpo" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "错误: 未找到 stage_b_orpo 的 checkpoint"
        exit 1
    fi
    CHECKPOINT_PATH="$LATEST_CHECKPOINT"
    CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
    echo "自动选择最新的 checkpoint: $CHECKPOINT_NAME"
else
    CHECKPOINT_PATH="$LLAMA_FACTORY_DIR/saves/stage_b_orpo/checkpoint-$1"
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "错误: checkpoint 不存在: $CHECKPOINT_PATH"
        exit 1
    fi
    CHECKPOINT_NAME="checkpoint-$1"
fi

echo "=========================================="
echo "生成提交文件"
echo "=========================================="
echo "基础模型: $BASE_MODEL_PATH"
echo "Adapter: $CHECKPOINT_PATH"
echo "输出文件: $PROJECT_DIR/submission.json"
echo "=========================================="

cd "$PROJECT_DIR"

# 运行生成脚本
python scripts/generate_submission.py \
  --base_model_path "$BASE_MODEL_PATH" \
  --adapter_path "$CHECKPOINT_PATH" \
  --samples_file data/final_dataset.json \
  --schema_file data/schema.json \
  --knowledge_file data/common_knowledge.md \
  --output submission.json

echo ""
echo "=========================================="
echo "生成完成！"
echo "提交文件: $PROJECT_DIR/submission.json"
echo "=========================================="

# 检查生成的文件
if [ -f "$PROJECT_DIR/submission.json" ]; then
    SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('submission.json'))))")
    echo "包含 $SAMPLE_COUNT 条SQL预测"
else
    echo "警告: submission.json 未生成"
    exit 1
fi

