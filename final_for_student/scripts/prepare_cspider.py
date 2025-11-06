#!/usr/bin/env python3
"""
CSpider 数据转换脚本
将 CSpider 数据集转换为 LLaMA-Factory 需要的 conversations 格式
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# 添加脚本目录到路径，以便导入 adapt_sql_to_starrocks
sys.path.insert(0, str(Path(__file__).parent))
from adapt_sql_to_starrocks import adapt_sql_to_starrocks


def load_gold_sqls(gold_sql_file: str) -> Dict[str, str]:
    """
    加载 train_gold.sql 文件，返回 {index: sql} 映射
    
    Args:
        gold_sql_file: train_gold.sql 文件路径
        
    Returns:
        索引到 SQL 的映射字典
    """
    gold_sqls = {}
    with open(gold_sql_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # 格式：SQL语句\t数据库名
            parts = line.split('\t')
            if len(parts) >= 2:
                sql = parts[0].strip()
                db_id = parts[1].strip()
                gold_sqls[idx] = sql
            else:
                # 如果没有 tab 分隔，整行都是 SQL
                gold_sqls[idx] = parts[0] if parts else line
    return gold_sqls


def load_tables(tables_file: str) -> Dict[str, Any]:
    """
    加载 tables.json 文件，返回数据库 schema 信息
    
    Args:
        tables_file: tables.json 文件路径
        
    Returns:
        数据库 ID 到 schema 的映射字典
    """
    with open(tables_file, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # 按 db_id 组织
    db_schemas = {}
    for item in tables_data:
        db_id = item.get('db_id')
        if db_id:
            if db_id not in db_schemas:
                db_schemas[db_id] = {
                    'table_names_original': item.get('table_names_original', []),
                    'table_names': item.get('table_names', []),
                    'column_names_original': item.get('column_names_original', []),
                    'column_names': item.get('column_names', []),
                    'foreign_keys': item.get('foreign_keys', []),
                    'primary_keys': item.get('primary_keys', []),
                }
    
    return db_schemas


def format_schema_summary(db_id: str, schema: Dict[str, Any]) -> str:
    """
    格式化数据库 schema 摘要
    
    Args:
        db_id: 数据库 ID
        schema: schema 信息字典
        
    Returns:
        schema 摘要字符串
    """
    lines = [f"数据库: {db_id}"]
    
    table_names = schema.get('table_names', [])
    if table_names:
        lines.append(f"表: {', '.join(table_names)}")
    
    # 可以添加更多 schema 信息，如列名、主键等
    # 这里先简单处理
    
    return "\n".join(lines)


def convert_cspider_to_conversations(
    train_json_file: str,
    gold_sql_file: str,
    tables_json_file: str,
    output_file: str,
    max_samples: int = None
) -> None:
    """
    将 CSpider 数据转换为 LLaMA-Factory conversations 格式
    
    Args:
        train_json_file: train.json 文件路径
        gold_sql_file: train_gold.sql 文件路径
        tables_json_file: tables.json 文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数（用于测试）
    """
    print(f"Loading gold SQLs from {gold_sql_file}...")
    gold_sqls = load_gold_sqls(gold_sql_file)
    print(f"Loaded {len(gold_sqls)} gold SQLs")
    
    print(f"Loading tables from {tables_json_file}...")
    db_schemas = load_tables(tables_json_file)
    print(f"Loaded schemas for {len(db_schemas)} databases")
    
    print(f"Loading train data from {train_json_file}...")
    with open(train_json_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} samples")
    
    conversations = []
    skipped = 0
    
    for idx, sample in enumerate(train_data):
        if max_samples and idx >= max_samples:
            break
        
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(train_data)}...")
        
        # 提取必要字段
        db_id = sample.get('db_id')
        question = sample.get('question', '')
        
        if not question or not db_id:
            skipped += 1
            continue
        
        # 获取对应的 SQL（从 gold_sqls 中）
        if idx >= len(gold_sqls):
            print(f"Warning: Sample {idx} has no corresponding gold SQL")
            skipped += 1
            continue
        
        original_sql = gold_sqls[idx]
        
        # 适配 SQL 到 StarRocks
        adapted_sql = adapt_sql_to_starrocks(original_sql)
        
        # 获取 schema 摘要（可选）
        schema_summary = ""
        if db_id in db_schemas:
            schema_summary = format_schema_summary(db_id, db_schemas[db_id])
        
        # 构建 user prompt
        user_prompt = f"【问题】{question}\n【db】{db_id}"
        if schema_summary:
            user_prompt += f"\n{schema_summary}"
        
        # 构建 conversations 格式
        conversation = {
            "conversations": [
                {
                    "role": "system",
                    "content": "你是 StarRocks 方言的 SQL 助手：统一用字符串日期；优先 CTE；禁用 FULL OUTER JOIN；空值用 COALESCE；字符串函数用 SUBSTR/LOCATE；输出字段名与题面一致。"
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": adapted_sql
                }
            ]
        }
        
        conversations.append(conversation)
    
    print(f"Converted {len(conversations)} samples, skipped {skipped}")
    
    # 保存为 JSON 文件
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Output saved to {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSpider data to LLaMA-Factory format')
    parser.add_argument('--train_json', type=str, required=True,
                        help='Path to train.json')
    parser.add_argument('--gold_sql', type=str, required=True,
                        help='Path to train_gold.sql')
    parser.add_argument('--tables_json', type=str, required=True,
                        help='Path to tables.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    convert_cspider_to_conversations(
        train_json_file=args.train_json,
        gold_sql_file=args.gold_sql,
        tables_json_file=args.tables_json,
        output_file=args.output,
        max_samples=args.max_samples
    )

