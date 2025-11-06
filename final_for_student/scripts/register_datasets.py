#!/usr/bin/env python3
"""
注册数据集到 LLaMA-Factory
将数据集配置添加到 LLaMA-Factory 的 dataset_info.json
"""

import json
import sys
from pathlib import Path


def register_datasets(
    datasets_config_file: str,
    llama_factory_dataset_info: str
) -> None:
    """
    注册数据集
    
    Args:
        datasets_config_file: 数据集配置文件路径（我们创建的 datasets.json）
        llama_factory_dataset_info: LLaMA-Factory 的 dataset_info.json 路径
    """
    # 加载我们的数据集配置
    with open(datasets_config_file, 'r', encoding='utf-8') as f:
        new_datasets = json.load(f)
    
    # 加载 LLaMA-Factory 的 dataset_info.json
    with open(llama_factory_dataset_info, 'r', encoding='utf-8') as f:
        existing_datasets = json.load(f)
    
    # 合并数据集
    for dataset_name, dataset_config in new_datasets.items():
        if dataset_name in existing_datasets:
            print(f"Warning: Dataset '{dataset_name}' already exists, updating...")
        existing_datasets[dataset_name] = dataset_config
    
    # 保存更新后的配置
    with open(llama_factory_dataset_info, 'w', encoding='utf-8') as f:
        json.dump(existing_datasets, f, ensure_ascii=False, indent=2)
    
    print(f"Registered {len(new_datasets)} datasets:")
    for name in new_datasets.keys():
        print(f"  - {name}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Register datasets to LLaMA-Factory')
    parser.add_argument('--datasets_config', type=str, 
                        default='/home/ubuntu/workspace/tencent/final_for_student/configs/datasets.json',
                        help='Dataset configuration file')
    parser.add_argument('--llama_factory_info', type=str,
                        default='/home/ubuntu/workspace/LLaMA-Factory/data/dataset_info.json',
                        help='LLaMA-Factory dataset_info.json path')
    
    args = parser.parse_args()
    
    register_datasets(
        datasets_config_file=args.datasets_config,
        llama_factory_dataset_info=args.llama_factory_info
    )

