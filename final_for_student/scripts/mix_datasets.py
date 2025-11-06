#!/usr/bin/env python3
"""
数据混合脚本
按 token 数将阶段 A（CSpider）和阶段 B（域内数据）按 60:40 混合
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import random


def count_tokens(text: str, tokenizer) -> int:
    """
    计算文本的 token 数
    
    Args:
        text: 文本字符串
        tokenizer: tokenizer 对象
        
    Returns:
        token 数
    """
    if not text:
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def load_tokenizer(model_path: str):
    """
    加载 Qwen tokenizer
    
    Args:
        model_path: 模型路径
        
    Returns:
        tokenizer 对象
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
    except ImportError:
        print("Error: transformers not installed. Please install it first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {e}")
        sys.exit(1)


def calculate_total_tokens(conversations: List[Dict], tokenizer) -> int:
    """
    计算一个 conversation 样本的总 token 数
    
    Args:
        conversations: conversation 字典列表
        tokenizer: tokenizer 对象
        
    Returns:
        总 token 数
    """
    total = 0
    for conv in conversations:
        role = conv.get('role', '')
        content = conv.get('content', '')
        # 计算 content 的 token 数，加上 role 等特殊标记的近似值
        total += count_tokens(content, tokenizer)
        # 添加角色标记的近似 token 数（如 <|im_start|>role 等）
        total += 5  # 粗略估计
    return total


def mix_datasets(
    stage_a_file: str,
    stage_b_file: str,
    output_file: str,
    ratio_a: float = 0.6,
    ratio_b: float = 0.4,
    model_path: str = None,
    shuffle: bool = True
) -> None:
    """
    按 token 数混合两个数据集
    
    Args:
        stage_a_file: 阶段 A 数据文件（CSpider）
        stage_b_file: 阶段 B 数据文件（域内数据）
        output_file: 输出文件路径
        ratio_a: 阶段 A 的比例（默认 0.6）
        ratio_b: 阶段 B 的比例（默认 0.4）
        model_path: 模型路径（用于 tokenizer）
        shuffle: 是否打乱数据
    """
    # 加载 tokenizer
    if model_path is None:
        # 默认使用 Qwen2.5 14B 模型路径
        model_path = "/home/ubuntu/workspace/tencent/models/Qwen_Qwen2.5-14B-Instruct"
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = load_tokenizer(model_path)
    
    # 加载数据
    print(f"Loading stage A data from {stage_a_file}...")
    with open(stage_a_file, 'r', encoding='utf-8') as f:
        stage_a_data = json.load(f)
    print(f"Loaded {len(stage_a_data)} samples from stage A")
    
    print(f"Loading stage B data from {stage_b_file}...")
    with open(stage_b_file, 'r', encoding='utf-8') as f:
        stage_b_data = json.load(f)
    print(f"Loaded {len(stage_b_data)} samples from stage B")
    
    # 计算每个样本的 token 数
    print("Calculating tokens for stage A samples...")
    stage_a_tokens = []
    for idx, sample in enumerate(stage_a_data):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(stage_a_data)}...")
        convs = sample.get('conversations', [])
        tokens = calculate_total_tokens(convs, tokenizer)
        stage_a_tokens.append((sample, tokens))
    
    print("Calculating tokens for stage B samples...")
    stage_b_tokens = []
    for idx, sample in enumerate(stage_b_data):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(stage_b_data)}...")
        convs = sample.get('conversations', [])
        tokens = calculate_total_tokens(convs, tokenizer)
        stage_b_tokens.append((sample, tokens))
    
    # 计算总 token 数
    total_a_tokens = sum(tokens for _, tokens in stage_a_tokens)
    total_b_tokens = sum(tokens for _, tokens in stage_b_tokens)
    
    print(f"\nStage A total tokens: {total_a_tokens:,}")
    print(f"Stage B total tokens: {total_b_tokens:,}")
    
    # 计算目标混合后的总 token 数（假设使用全部数据）
    # 实际混合时，我们需要确保比例是 ratio_a:ratio_b
    
    # 计算需要的 token 数
    # 如果我们使用所有 stage B 数据（ratio_b 比例），那么 stage A 应该用多少？
    # total_mixed = total_b_tokens / ratio_b  (如果我们用全部 B)
    # total_a_needed = total_mixed * ratio_a
    
    # 更简单的策略：采样 stage A 和 stage B 的样本，使得比例接近 ratio_a:ratio_b
    # 我们可以使用全部 B，然后采样 A 使得比例匹配
    
    # 目标：stage_a_used_tokens / stage_b_used_tokens = ratio_a / ratio_b
    # 如果我们用全部 B，则 stage_a_used_tokens = total_b_tokens * ratio_a / ratio_b
    
    target_a_tokens = int(total_b_tokens * ratio_a / ratio_b)
    
    print(f"\nTarget Stage A tokens (to match ratio): {target_a_tokens:,}")
    
    # 采样 stage A 样本以达到目标 token 数
    if shuffle:
        random.shuffle(stage_a_tokens)
    
    selected_a_samples = []
    current_a_tokens = 0
    
    for sample, tokens in stage_a_tokens:
        if current_a_tokens + tokens <= target_a_tokens:
            selected_a_samples.append(sample)
            current_a_tokens += tokens
            if current_a_tokens >= target_a_tokens:
                break
    
    print(f"Selected {len(selected_a_samples)} samples from stage A ({current_a_tokens:,} tokens)")
    print(f"Using {len(stage_b_data)} samples from stage B ({total_b_tokens:,} tokens)")
    
    # 混合数据
    mixed_data = []
    
    # 添加 stage A 样本（带标记）
    for sample in selected_a_samples:
        # 可以添加元数据标记
        sample_copy = json.loads(json.dumps(sample))  # 深拷贝
        # sample_copy['meta'] = {'stage': 'A', 'source': 'CSpider'}
        mixed_data.append(sample_copy)
    
    # 添加 stage B 样本（带标记）
    for sample in stage_b_data:
        sample_copy = json.loads(json.dumps(sample))  # 深拷贝
        # sample_copy['meta'] = {'stage': 'B', 'source': 'domain'}
        mixed_data.append(sample_copy)
    
    # 打乱混合后的数据
    if shuffle:
        random.shuffle(mixed_data)
    
    # 计算最终 token 比例
    final_a_tokens = sum(calculate_total_tokens(s.get('conversations', []), tokenizer) 
                        for s in selected_a_samples)
    final_b_tokens = total_b_tokens
    
    total_final_tokens = final_a_tokens + final_b_tokens
    final_ratio_a = final_a_tokens / total_final_tokens if total_final_tokens > 0 else 0
    final_ratio_b = final_b_tokens / total_final_tokens if total_final_tokens > 0 else 0
    
    print(f"\nFinal mix:")
    print(f"  Stage A: {final_a_tokens:,} tokens ({final_ratio_a*100:.1f}%)")
    print(f"  Stage B: {final_b_tokens:,} tokens ({final_ratio_b*100:.1f}%)")
    print(f"  Total: {total_final_tokens:,} tokens")
    print(f"  Total samples: {len(mixed_data)}")
    
    # 保存混合后的数据
    print(f"\nSaving mixed data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mixed_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mix datasets by token ratio')
    parser.add_argument('--stage_a', type=str, required=True,
                        help='Stage A data file (CSpider)')
    parser.add_argument('--stage_b', type=str, required=True,
                        help='Stage B data file (domain data)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--ratio_a', type=float, default=0.6,
                        help='Ratio of stage A (default: 0.6)')
    parser.add_argument('--ratio_b', type=float, default=0.4,
                        help='Ratio of stage B (default: 0.4)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model path for tokenizer (default: Qwen2.5-14B)')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle the mixed data')
    
    args = parser.parse_args()
    
    # 验证比例
    if abs(args.ratio_a + args.ratio_b - 1.0) > 1e-6:
        print(f"Warning: ratio_a + ratio_b = {args.ratio_a + args.ratio_b}, not equal to 1.0")
    
    mix_datasets(
        stage_a_file=args.stage_a,
        stage_b_file=args.stage_b,
        output_file=args.output,
        ratio_a=args.ratio_a,
        ratio_b=args.ratio_b,
        model_path=args.model_path,
        shuffle=not args.no_shuffle
    )

