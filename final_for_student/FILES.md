# StarRocks NL-to-SQL 项目文件清单

## 📁 需要提交的文件

### 核心代码和配置
- `scripts/` - 所有脚本文件（数据准备、评估、提交等）
- `configs/` - 训练配置文件（YAML 和 JSON）
- `docs/` - 文档目录（SQL 适配规则等）
- `README.md` - 项目说明文档

### 数据文件（选择性提交）
- `data/schema.json` - 数据库 schema（必需）
- `data/final_dataset.json` - 最终评估数据集（必需）
- `data/common_knowledge.md` - 公共业务知识（必需）
- `data/sft_conversation.json` - 域内原始数据（建议提交）
- `data/success_samples.jsonl` - 评测样本（建议提交）

### 示例和参考
- `upload_example/` - 提交格式示例（如果有）
- `code/` - 代码目录（如果有必要）

## 🚫 应该忽略的文件（已在 .gitignore 中）

### 训练输出
- `saves/` - 训练 checkpoint（可能很大，通常不提交）
- `checkpoints/` - 其他 checkpoint 目录

### 虚拟环境和缓存
- `venv/` - Python 虚拟环境
- `__pycache__/` - Python 缓存文件
- `*.pyc`, `*.pyo` - Python 编译文件

### 结果和输出文件
- `result/` - 评估结果
- `submission.json` - 提交文件（每次运行生成）
- `tmp/` - 临时文件

### 系统文件
- `.DS_Store` - macOS 系统文件
- `__MACOSX/` - macOS 压缩文件元数据
- `*.swp`, `*.swo` - 编辑器临时文件

### 大型文件
- `data/cspider_train.json` - 6.2M（如果太大，考虑使用 Git LFS）
- `data/insert_sql.json` - 6.3M（如果太大，考虑使用 Git LFS）
- `*.zip` - 压缩文件

## 📝 建议

1. **大型数据文件**：如果 `cspider_train.json` 和 `insert_sql.json` 很大，考虑：
   - 使用 Git LFS（Large File Storage）
   - 或者提供下载链接，不直接提交到仓库

2. **训练输出**：`saves/` 目录通常很大，建议：
   - 不提交到仓库
   - 在 README 中说明如何训练
   - 或者提供模型下载链接

3. **临时文件**：`tmp/` 和 `result/` 目录包含运行时生成的文件，不应提交

4. **提交脚本**：`submission.json` 是每次运行生成的，不应提交

## 🔍 检查清单

在提交前，确认：
- [ ] 已创建 `.gitignore` 文件
- [ ] 虚拟环境已忽略
- [ ] 训练输出已忽略
- [ ] 临时文件已忽略
- [ ] 大型数据文件已处理（LFS 或忽略）
- [ ] README.md 已更新
- [ ] 代码注释清晰

