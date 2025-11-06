# StarRocks NL-to-SQL 翻译项目

## 📋 项目概述

本项目旨在实现稳定的自然语言到 SQL 的翻译系统，专门针对 **StarRocks** 数据库方言。目标是在 StarRocks 上生成**可执行且口径正确**的 SQL 查询。

### 核心要求

- **可执行性**：生成的 SQL 必须符合 StarRocks 语法规范，能够成功执行
- **口径正确性**：SQL 的业务逻辑必须符合题面要求，包括字段名、日期格式、业务口径等

---

## 🎯 技术方案

### 双阶段 QLoRA 微调策略

采用**两阶段渐进式训练**策略，从通用能力到域内对齐：

#### Stage A：通用能力训练（SFT）
- **数据集**：CSpider（中文 SQL 数据集）
- **目标**：学习复杂 SQL 结构
  - CTE（WITH 子句）
  - 多表 JOIN
  - 子查询
  - 窗口函数
  - HAVING 子句
- **适配**：将 CSpider 的 SQL 适配为 StarRocks 方言

#### Stage B：域内对齐训练（SFT + ORPO）
- **数据集**：官方 72 条样本的程序化增广（扩至 200-500 条）
- **目标**：锁定 StarRocks 方言和业务口径
  - 日期格式：统一使用字符串 `'YYYYMMDD'`
  - 业务口径：如 `saccounttype='-100'`、`suseridtype in ('qq','wxid')`
  - 输出字段名：与题面一致
  - 函数适配：`COALESCE`、`SUBSTR`、`LOCATE` 等
- **方法**：
  1. **SFT**：域内数据的监督微调
  2. **ORPO**（可选）：使用 chosen/rejected 对进行偏好优化

### 数据配比

- **CSpider : 域内增广 = 60% : 40%**（按 token 数计算）

### 模型与框架

- **基础模型**：Qwen/Qwen2.5-14B-Instruct
- **训练框架**：LLaMA-Factory
- **量化方法**：QLoRA（4-bit 量化）
- **LoRA 参数**：
  - `lora_rank: 16`
  - `lora_alpha: 32`
  - `lora_dropout: 0.05`
  - `lora_target: all`

---

## 🛠️ 环境构建

### 系统要求

- **操作系统**：Linux（推荐 Ubuntu 20.04+）
- **Python**：3.10+（推荐 3.10 或 3.12）
- **CUDA**：11.6+（推荐 12.2）
- **GPU**：V100 32GB 或更高（QLoRA 4-bit 约需 12GB 显存）

### Python 环境搭建

#### 1. 创建虚拟环境

```bash
# 在项目根目录创建虚拟环境
cd /home/ubuntu/workspace/tencent
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

#### 2. 安装 PyTorch

```bash
# 根据 CUDA 版本选择（以 CUDA 12.1 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. 安装 LLaMA-Factory

```bash
cd /home/ubuntu/workspace/LLaMA-Factory
pip install -e ".[torch,bitsandbytes]"
```

#### 4. 安装其他依赖

```bash
# 安装 StarRocks 数据库连接库
pip install PyMySQL

# 安装其他必要库
pip install transformers>=4.49.0
pip install peft>=0.14.0
pip install accelerate>=1.3.0
pip install datasets>=2.16.0
pip install trl>=0.8.6
pip install safetensors
pip install sentencepiece
pip install tiktoken
```

#### 5. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

### 目录结构

```
tencent/
├── final_for_student/          # 项目主目录
│   ├── configs/                # 训练配置文件
│   │   ├── stage_a_sft.yaml
│   │   ├── stage_b_sft.yaml
│   │   └── stage_b_orpo.yaml
│   ├── data/                   # 数据目录
│   │   ├── cspider_train.json  # CSpider 训练数据
│   │   ├── sft_conversation.json # 域内 SFT 数据
│   │   ├── final_dataset.json  # 最终评估数据集（101条）
│   │   ├── schema.json         # 数据库 schema
│   │   └── common_knowledge.md # 公共业务知识
│   ├── scripts/                # 脚本目录
│   │   ├── adapt_sql_to_starrocks.py  # SQL 适配脚本
│   │   ├── prepare_cspider.py         # CSpider 数据准备
│   │   ├── prepare_domain_data.py     # 域内数据准备
│   │   ├── mix_datasets.py            # 数据集混合
│   │   ├── register_datasets.py      # 数据集注册
│   │   ├── evaluate.py                # 评估脚本
│   │   ├── generate_submission.py     # 提交生成脚本
│   │   └── run_submission.sh          # 提交便捷脚本
│   ├── saves/                   # 训练输出目录（在 LLaMA-Factory 中）
│   └── docs/                    # 文档目录
│       └── sql_adaptation_rules.md    # SQL 适配规则文档
├── LLaMA-Factory/              # LLaMA-Factory 框架
└── models/                     # 模型目录
    └── Qwen_Qwen2.5-14B-Instruct/
```

---

## 📊 数据准备

### 1. CSpider 数据准备

将 CSpider 原始数据转换为 LLaMA-Factory 的 **sharegpt** 格式，并应用 StarRocks SQL 适配。

**数据格式说明**：
- 虽然使用 sharegpt 格式，但我们的任务是**单轮对话**（每个样本：system → user → assistant）
- Sharegpt 格式也支持单轮对话，LLaMA-Factory 内部会将其转换为统一的 messages 格式
- 通过 `tags` 配置将 `role`/`content` 字段映射到 LLaMA-Factory 的内部格式
- 理论上也可以使用 alpaca 格式，但 sharegpt 格式已经足够且更通用

```bash
cd /home/ubuntu/workspace/tencent/final_for_student
source /home/ubuntu/workspace/tencent/.venv/bin/activate

python scripts/prepare_cspider.py \
  --train_json /path/to/CSpider/train.json \
  --gold_sql /path/to/CSpider/train_gold.sql \
  --tables_json /path/to/CSpider/tables.json \
  --output data/cspider_train.json
```

### 2. 域内数据准备

准备域内 SFT 和 ORPO 数据：

```bash
python scripts/prepare_domain_data.py \
  --input data/sft_conversation.json \
  --output_sft data/domain_sft.json \
  --output_orpo data/domain_orpo.json
```

### 3. 数据集混合

按 token 比例混合 CSpider 和域内数据（60:40）：

```bash
python scripts/mix_datasets.py \
  --cspider_path data/cspider_train.json \
  --domain_path data/domain_sft.json \
  --output data/cspider_domain_mixed.json \
  --tokenizer_path /home/ubuntu/workspace/tencent/models/Qwen_Qwen2.5-14B-Instruct \
  --cspider_ratio 0.6
```

### 4. 注册数据集到 LLaMA-Factory

```bash
python scripts/register_datasets.py \
  --llama_factory_data_path /home/ubuntu/workspace/LLaMA-Factory/data \
  --custom_datasets_config_path configs/datasets.json
```

---

## 🚀 训练流程

### Stage A：通用能力训练（SFT）

```bash
cd /home/ubuntu/workspace/LLaMA-Factory
source /home/ubuntu/workspace/tencent/.venv/bin/activate

llamafactory-cli train /home/ubuntu/workspace/tencent/final_for_student/configs/stage_a_sft.yaml
```

**关键参数：**
- `cutoff_len: 2048` - 序列截断长度（数据最大长度 617）
- `packing: true` - 样本打包，减少 padding
- `group_by_length: true` - 按长度分桶
- `gradient_accumulation_steps: 32` - 梯度累积
- `learning_rate: 1.0e-4` - 学习率
- `num_train_epochs: 1.0` - 训练 1 个 epoch

**输出：** `LLaMA-Factory/saves/stage_a_sft/`

### Stage B：域内对齐训练（SFT）

```bash
llamafactory-cli train /home/ubuntu/workspace/tencent/final_for_student/configs/stage_b_sft.yaml
```

**关键参数：**
- `adapter_name_or_path: saves/stage_a_sft` - 从 Stage A 继续训练
- `cutoff_len: 3072` - 域内数据较长，保持 3072
- `learning_rate: 5.0e-5` - 学习率降低
- `num_train_epochs: 1.0` - 训练 1 个 epoch

**输出：** `LLaMA-Factory/saves/stage_b_sft/`

### Stage B：域内对齐训练（ORPO，可选）

```bash
llamafactory-cli train /home/ubuntu/workspace/tencent/final_for_student/configs/stage_b_orpo.yaml
```

**关键参数：**
- `adapter_name_or_path: saves/stage_b_sft` - 从 Stage B SFT 继续训练
- `stage: dpo` - 使用 DPO stage
- `pref_loss: orpo` - 使用 ORPO loss
- `pref_beta: 0.1` - ORPO beta 参数
- `packing: false` - ORPO 通常不启用 packing
- `learning_rate: 1.0e-5` - 进一步降低学习率
- `num_train_epochs: 1.5` - 训练 1.5 个 epoch（可在 1.0 后继续）

**输出：** `LLaMA-Factory/saves/stage_b_orpo/`

**续训：** 重复执行上述命令会自动从最新 checkpoint 继续训练（`overwrite_output_dir: false`，`resume_from_checkpoint: null`）

---

## 📈 评估

### 评估脚本

使用评估脚本测试模型的可执行率和正确率：

```bash
cd /home/ubuntu/workspace/tencent/final_for_student
source /home/ubuntu/workspace/tencent/.venv/bin/activate

python scripts/evaluate.py \
  --base_model_path /home/ubuntu/workspace/tencent/models/Qwen_Qwen2.5-14B-Instruct \
  --adapter_path /home/ubuntu/workspace/LLaMA-Factory/saves/stage_b_orpo/checkpoint-12 \
  --samples_file data/success_samples.jsonl \
  --schema_file data/schema.json \
  --db_host 127.0.0.1 \
  --db_port 9030 \
  --db_user root \
  --db_password "" \
  --db_name testDB \
  --output evaluation_result.json \
  --save_predictions predictions.json
```

### 评估指标

- **可执行率**：生成的 SQL 能够成功执行的百分比
- **正确率**：执行结果与预期结果匹配的百分比（需要提供 ground truth）

---

## 📤 提交

### 生成提交文件

使用提交脚本生成比赛提交文件：

```bash
# 方法1：使用便捷脚本（自动查找最新 checkpoint）
bash scripts/run_submission.sh

# 方法2：指定 checkpoint 编号
bash scripts/run_submission.sh 12

# 方法3：直接使用 Python 脚本
python scripts/generate_submission.py \
  --base_model_path /home/ubuntu/workspace/tencent/models/Qwen_Qwen2.5-14B-Instruct \
  --adapter_path /home/ubuntu/workspace/LLaMA-Factory/saves/stage_b_orpo/checkpoint-12 \
  --samples_file data/final_dataset.json \
  --schema_file data/schema.json \
  --knowledge_file data/common_knowledge.md \
  --output submission.json
```

**输出格式：**

```json
[
  {
    "sql_id": "sql_1",
    "sql": "SELECT ..."
  },
  {
    "sql_id": "sql_2",
    "sql": "SELECT ..."
  },
  ...
]
```

### 注意事项

1. **数据准备**：确保已通过 `sql_exe.py` 执行 `insert_sql.json` 完成数据插入
2. **题目数量**：`final_dataset.json` 包含 101 条题目，比赛系统会自动筛选 86 道
3. **提交格式**：输出文件必须包含所有题目的 `sql_id` 和 `sql` 字段

---

## ⚙️ 训练加速技巧

### 已应用的优化

1. **序列截断**：`cutoff_len` 从 3072 降至 2048（Stage A）
2. **样本打包**：`packing: true` + `group_by_length: true`
3. **梯度累积**：`gradient_accumulation_steps: 32`（Stage A）
4. **减少 I/O**：`save_steps: 1000`，`eval_steps: 1000`，`logging_steps: 100`
5. **缓存优化**：`overwrite_cache: false`，使用预处理缓存

### 进一步优化建议

- 降低 `lora_rank` 从 16 到 8（速度提升，效果可能小幅下降）
- 只训练注意力层：`lora_target: q_proj,k_proj,v_proj,o_proj`
- 减少训练轮数：先用 0.5-1 epoch 得到 baseline，再用 ORPO 精修

---

## 📝 SQL 适配规则

StarRocks 方言的 SQL 适配规则详见：`docs/sql_adaptation_rules.md`

**关键规则：**
- **日期格式**：统一使用字符串 `'YYYYMMDD'`
- **空值处理**：使用 `COALESCE(x, y)`
- **字符串函数**：使用 `SUBSTR(s, pos, len)` 和 `LOCATE(substr, s)`
- **禁用 FULL OUTER JOIN**：需改写为 `LEFT JOIN ... UNION ALL ...`
- **CTE**：优先使用 `WITH` 子句
- **别名**：统一使用小写蛇形命名（`snake_case`）

---

## 🐛 常见问题

### 1. 训练中断后如何续训？

重复执行相同的训练命令即可，LLaMA-Factory 会自动从最新 checkpoint 继续（`resume_from_checkpoint: null`）。

### 2. 如何选择 checkpoint？

评估不同 checkpoint 的性能，选择可执行率和正确率最高的。

### 3. 提交脚本中断后如何继续？

**当前版本不支持断点续传**，建议：
- 使用 `nohup` 后台运行
- 或使用 `screen`/`tmux` 会话
- 或分批处理（使用 `--max_samples` 参数）

### 4. 显存不足怎么办？

- 降低 `cutoff_len`
- 降低 `gradient_accumulation_steps`（但需要相应提高 `learning_rate`）
- 降低 `lora_rank`

---

## 📚 参考资料

- [LLaMA-Factory 文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 模型文档](https://github.com/QwenLM/Qwen2.5)
- [StarRocks SQL 参考](https://docs.starrocks.io/docs/sql-reference/sql-statements/)

---

## 📄 许可证

本项目遵循相关开源许可证。
