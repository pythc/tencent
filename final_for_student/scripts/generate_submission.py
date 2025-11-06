#!/usr/bin/env python3
"""
生成提交结果脚本
对 final_dataset.json 中的题目生成 SQL，并保存为提交格式
"""

import json
import argparse
import sys
import re
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 导入评估脚本中的函数（如果可用）
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
try:
    from evaluate import load_schema, extract_schema_summary, extract_sql_from_text
except ImportError:
    # 如果导入失败，直接定义这些函数
    def load_schema(schema_file: str) -> Dict[str, Dict]:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_list = json.load(f)
        schema_dict = {}
        for table_info in schema_list:
            table_name = table_info.get('table_name')
            if table_name:
                schema_dict[table_name] = table_info
        return schema_dict
    
    def extract_schema_summary(schema_dict: Dict[str, Dict], table_list: List[str], knowledge: str = "") -> str:
        schema_parts = []
        for table_name in table_list:
            if table_name in schema_dict:
                table_info = schema_dict[table_name]
                table_desc = table_info.get('table_description', table_name)
                schema_parts.append(f"表: {table_name} ({table_desc})")
                columns = table_info.get('columns', [])
                if columns:
                    col_info = []
                    for col in columns[:20]:
                        col_name = col.get('col', '')
                        col_type = col.get('type', '')
                        col_desc = col.get('description', '')
                        col_str = f"{col_name}({col_type})"
                        if col_desc:
                            col_str += f": {col_desc}"
                        col_info.append(col_str)
                    if len(columns) > 20:
                        col_info.append(f"... 还有 {len(columns) - 20} 个列")
                    schema_parts.append(f"  列: {', '.join(col_info)}")
            else:
                schema_parts.append(f"表: {table_name} (未找到schema信息)")
        schema_text = "\n".join(schema_parts)
        if knowledge:
            schema_text += f"\n\n业务知识:\n{knowledge}"
        return schema_text
    
    def extract_sql_from_text(text: str) -> str:
        text = text.strip()
        sql_pattern = re.compile(r'```(?:sql)?\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
        match = sql_pattern.search(text)
        if match:
            return match.group(1).strip()
        sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        for keyword in sql_keywords:
            idx = text.upper().find(keyword)
            if idx != -1:
                sql_text = text[idx:].strip()
                sql_text = re.sub(r'<\|im_end\|>.*$', '', sql_text, flags=re.DOTALL)
                return sql_text.strip()
        text = re.sub(r'<\|im_end\|>.*$', '', text, flags=re.DOTALL)
        return text.strip()


class SQLModelInference:
    """
    使用 transformers + peft 进行SQL生成的推理类
    """
    def __init__(self, base_model_path: str, adapter_path: str, device: str = "auto"):
        """
        初始化模型
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: LoRA adapter路径
            device: 设备（auto/cuda/cpu）
        """
        print(f"Loading base model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA adapter
        if adapter_path:
            # 处理相对路径和绝对路径
            adapter_path_abs = Path(adapter_path)
            if not adapter_path_abs.is_absolute():
                # 如果是相对路径，尝试多个可能的位置
                possible_paths = [
                    Path(__file__).parent.parent / adapter_path,  # 相对于脚本目录
                    Path.cwd() / adapter_path,  # 相对于当前工作目录
                    Path("/home/ubuntu/workspace/LLaMA-Factory") / adapter_path,  # LLaMA-Factory 目录
                ]
                for path in possible_paths:
                    if path.exists() and (path / "adapter_config.json").exists():
                        adapter_path_abs = path
                        break
                else:
                    # 如果所有路径都不存在，使用原始路径（让 PeftModel 报错）
                    adapter_path_abs = Path(adapter_path)
            else:
                adapter_path_abs = Path(adapter_path)
            
            if not adapter_path_abs.exists():
                raise FileNotFoundError(f"Adapter path not found: {adapter_path_abs}")
            if not (adapter_path_abs / "adapter_config.json").exists():
                raise FileNotFoundError(f"Adapter config not found at: {adapter_path_abs}")
            
            print(f"Loading adapter from {adapter_path_abs}...")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path_abs))
            # 不合并adapter，保持LoRA模式以节省显存
        
        self.model.eval()
        
        # 系统提示（与训练时保持一致）
        self.system_prompt = (
            "你是 StarRocks 方言的 SQL 助手：统一用字符串日期；优先 CTE；禁用 FULL OUTER JOIN；"
            "空值用 COALESCE；字符串函数用 SUBSTR/LOCATE；输出字段名与题面一致。"
        )
        
        print("Model loaded successfully.")
    
    def generate_sql(self, question: str, schema_summary: str, max_new_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        生成SQL
        
        Args:
            question: 自然语言问题
            schema_summary: schema摘要
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top_p采样
            
        Returns:
            生成的SQL字符串
        """
        # 构造用户输入（与训练时格式一致）
        user_content = f"【问题】{question}\n【db】testDB"
        if schema_summary:
            user_content += f"\n{schema_summary}"
        
        # 构造消息格式（Qwen模板）
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 使用tokenizer的apply_chat_template构造prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        # 解码
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 提取SQL
        sql = extract_sql_from_text(generated_text)
        
        return sql


def load_common_knowledge(knowledge_file: str) -> str:
    """
    加载公共知识文件
    
    Args:
        knowledge_file: common_knowledge.md 文件路径
        
    Returns:
        知识内容字符串
    """
    try:
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {knowledge_file} not found, skipping common knowledge.")
        return ""


def generate_submission(
    base_model_path: str,
    adapter_path: str,
    samples_file: str,
    schema_file: str,
    output_file: str,
    knowledge_file: Optional[str] = None,
    max_samples: Optional[int] = None
) -> None:
    """
    生成提交结果
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: LoRA adapter路径
        samples_file: final_dataset.json 文件路径
        schema_file: schema.json 文件路径
        output_file: 输出文件路径（提交格式）
        knowledge_file: common_knowledge.md 文件路径（可选）
        max_samples: 最大样本数（用于测试）
    """
    print(f"Loading samples from {samples_file}...")
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if max_samples:
        samples = samples[:max_samples]
    print(f"Loaded {len(samples)} samples")
    
    # 加载schema
    print(f"Loading schema from {schema_file}...")
    schema_dict = load_schema(schema_file)
    print(f"Loaded schema for {len(schema_dict)} tables")
    
    # 加载公共知识
    common_knowledge = ""
    if knowledge_file:
        print(f"Loading common knowledge from {knowledge_file}...")
        common_knowledge = load_common_knowledge(knowledge_file)
    
    # 加载模型
    print("\nLoading model...")
    model_inference = SQLModelInference(base_model_path, adapter_path)
    
    # 生成SQL
    submission_results = []
    print("\nGenerating SQL for all samples...")
    
    for idx, sample in enumerate(samples):
        if idx % 10 == 0:
            print(f"  Processing {idx + 1}/{len(samples)}...")
        
        sql_id = sample.get('sql_id', f'sample_{idx}')
        question = sample.get('question', '')
        table_list = sample.get('table_list', [])
        knowledge = sample.get('knowledge', '')
        
        # 合并公共知识和题目知识
        full_knowledge = knowledge
        if common_knowledge:
            if full_knowledge:
                full_knowledge = f"{common_knowledge}\n\n{full_knowledge}"
            else:
                full_knowledge = common_knowledge
        
        # 提取schema摘要
        schema_summary = extract_schema_summary(schema_dict, table_list, full_knowledge)
        
        # 生成SQL
        try:
            generated_sql = model_inference.generate_sql(
                question=question,
                schema_summary=schema_summary,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            
            submission_results.append({
                "sql_id": sql_id,
                "sql": generated_sql
            })
            
        except Exception as e:
            error_msg = f"Model inference error: {str(e)}"
            print(f"  Error generating SQL for {sql_id}: {error_msg}")
            # 即使出错也保存一个空SQL，避免遗漏题目
            submission_results.append({
                "sql_id": sql_id,
                "sql": ""
            })
    
    # 保存提交结果
    print(f"\nSaving submission results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submission_results, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully generated {len(submission_results)} SQL predictions.")
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SQL submission file for competition')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Base model path (e.g., /path/to/Qwen_Qwen2.5-14B-Instruct)')
    parser.add_argument('--adapter_path', type=str, required=True,
                        help='LoRA adapter path (e.g., /path/to/stage_b_orpo/checkpoint-12)')
    parser.add_argument('--samples_file', type=str, default='data/final_dataset.json',
                        help='Path to final_dataset.json (default: data/final_dataset.json)')
    parser.add_argument('--schema_file', type=str, default='data/schema.json',
                        help='Path to schema.json (default: data/schema.json)')
    parser.add_argument('--knowledge_file', type=str, default='data/common_knowledge.md',
                        help='Path to common_knowledge.md (default: data/common_knowledge.md)')
    parser.add_argument('--output', type=str, default='submission.json',
                        help='Output submission file (default: submission.json)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    generate_submission(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        samples_file=args.samples_file,
        schema_file=args.schema_file,
        output_file=args.output,
        knowledge_file=args.knowledge_file,
        max_samples=args.max_samples
    )

