# Git 提交命令

## 快速提交

```bash
cd /home/ubuntu/workspace/tencent/final_for_student

# 1. 查看状态
git status

# 2. 添加所有文件（.gitignore 会自动过滤）
git add .

# 3. 查看将要提交的文件（确认无误）
git status

# 4. 提交
git commit -m "feat: 完成 StarRocks NL-to-SQL 双阶段 QLoRA 训练项目

- 添加数据准备脚本（CSpider、域内数据）
- 添加训练配置文件（Stage A/B SFT、ORPO）
- 添加评估和提交脚本
- 添加完整的项目文档
- 添加 .gitignore 和文件清单"

# 5. 推送到远程
git push origin main
```

## 分步提交（推荐）

```bash
cd /home/ubuntu/workspace/tencent/final_for_student

# 1. 查看当前状态
git status

# 2. 查看会被忽略的文件（确认 .gitignore 生效）
git status --ignored

# 3. 添加核心文件
git add .gitignore
git add README.md
git add FILES.md
git add scripts/
git add configs/
git add docs/

# 4. 添加数据文件（根据需要选择）
git add data/schema.json
git add data/final_dataset.json
git add data/common_knowledge.md
git add data/sft_conversation.json
git add data/success_samples.jsonl

# 5. 如果数据文件太大，可以跳过或使用 Git LFS
# git add data/cspider_train.json  # 6.2M
# git add data/insert_sql.json     # 6.3M

# 6. 查看将要提交的文件
git status

# 7. 提交
git commit -m "feat: 完成 StarRocks NL-to-SQL 双阶段 QLoRA 训练项目"

# 8. 推送
git push origin main
```

## 检查清单

提交前确认：
- [ ] `.gitignore` 已创建并生效
- [ ] 虚拟环境 `venv/` 不会被提交
- [ ] 训练输出 `saves/` 不会被提交
- [ ] 临时文件 `tmp/`, `result/` 不会被提交
- [ ] 大型数据文件已处理（LFS 或忽略）

## 如果遇到问题

### 文件太大无法推送
```bash
# 使用 Git LFS（如果文件 > 100MB）
git lfs install
git lfs track "*.json"
git add .gitattributes
git add data/cspider_train.json
git commit -m "Add large data files with Git LFS"
```

### 撤销最后一次提交
```bash
git reset --soft HEAD~1  # 保留更改
git reset --hard HEAD~1  # 丢弃更改
```

### 查看提交历史
```bash
git log --oneline
```

