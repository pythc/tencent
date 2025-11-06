#!/usr/bin/env python3
"""
准备域内数据（阶段 B）
将 sft_conversation.json 转换为 LLaMA-Factory 格式，并分离出可用于 ORPO 的数据
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def prepare_domain_data(
    input_file: str,
    output_sft_file: str,
    output_orpo_file: str = None
) -> None:
    """
    准备域内数据
    
    Args:
        input_file: sft_conversation.json 文件路径
        output_sft_file: 输出 SFT 数据文件路径
        output_orpo_file: 输出 ORPO 数据文件路径（可选）
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # sft_conversation.json 是每行一个 JSON 对象
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(data)} samples")
    
    # 转换为 LLaMA-Factory 格式
    sft_data = []
    orpo_data = []
    
    for item in data:
        conversations = item.get('conversations', [])
        rejected = item.get('rejected')
        
        # 基础 conversation 格式已经符合要求
        conv_item = {
            'conversations': conversations
        }
        sft_data.append(conv_item)
        
        # 如果有 rejected 字段，可以用于 ORPO
        # ORPO 格式需要：conversations（历史对话）+ chosen（优质回答）+ rejected（劣质回答）
        if rejected and output_orpo_file:
            # 从 conversations 中提取最后一个 assistant 的回答作为 chosen
            chosen_content = None
            conversations_without_assistant = []
            
            for conv in conversations:
                if conv.get('role') == 'assistant':
                    chosen_content = conv.get('content', '')
                else:
                    conversations_without_assistant.append(conv)
            
            if chosen_content:
                # ORPO 格式：chosen 和 rejected 都应该是对象格式
                # 使用 role/content 格式（因为我们设置了 tags）
                orpo_item = {
                    'conversations': conversations_without_assistant,
                    'chosen': {
                        'role': 'assistant',  # 对应 role_tag
                        'content': chosen_content  # 对应 content_tag
                    },
                    'rejected': {
                        'role': 'assistant',  # rejected 也应该是对象格式
                        'content': rejected if isinstance(rejected, str) else str(rejected)
                    }
                }
                orpo_data.append(orpo_item)
    
    # 保存 SFT 数据
    print(f"Saving SFT data to {output_sft_file}...")
    with open(output_sft_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(sft_data)} SFT samples")
    
    # 保存 ORPO 数据（如果有）
    if output_orpo_file and orpo_data:
        print(f"Saving ORPO data to {output_orpo_file}...")
        with open(output_orpo_file, 'w', encoding='utf-8') as f:
            json.dump(orpo_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(orpo_data)} ORPO samples")
    elif output_orpo_file:
        print("No ORPO data found (no rejected field)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare domain data for training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input sft_conversation.json file')
    parser.add_argument('--output_sft', type=str, required=True,
                        help='Output SFT data file')
    parser.add_argument('--output_orpo', type=str, default=None,
                        help='Output ORPO data file (optional)')
    
    args = parser.parse_args()
    
    prepare_domain_data(
        input_file=args.input,
        output_sft_file=args.output_sft,
        output_orpo_file=args.output_orpo
    )

