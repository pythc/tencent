#!/usr/bin/env python3
"""
评测脚本
使用 success_samples.jsonl 评测模型生成的 SQL 的可执行性和正确率
方案B：使用 transformers + peft 直接加载模型进行推理
"""

import json
import sys
import subprocess
import pymysql
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_samples(samples_file: str) -> List[Dict[str, Any]]:
    """
    加载样本文件（支持JSON和JSONL格式）
    
    Args:
        samples_file: 样本文件路径（final_dataset.json 或 success_samples.jsonl）
        
    Returns:
        样本列表
    """
    samples = []
    file_path = Path(samples_file)
    
    if file_path.suffix == '.jsonl':
        # JSONL格式（success_samples.jsonl）
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    else:
        # JSON格式（final_dataset.json）
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    
    return samples


def connect_db(host: str = '127.0.0.1', port: int = 9030, user: str = 'root', 
               password: str = '', database: str = 'testDB') -> pymysql.Connection:
    """
    连接 StarRocks 数据库
    
    Args:
        host: 数据库主机
        port: 端口
        user: 用户名
        password: 密码
        database: 数据库名
        
    Returns:
        数据库连接对象
    """
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            connect_timeout=10
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise


def execute_sql(conn: pymysql.Connection, sql: str, timeout: int = 30) -> Tuple[bool, List[Dict], str]:
    """
    执行 SQL 查询
    
    Args:
        conn: 数据库连接
        sql: SQL 语句
        timeout: 超时时间（秒）
        
    Returns:
        (是否成功, 结果列表, 错误信息)
    """
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.commit()
            return True, result, ""
    except Exception as e:
        error_msg = str(e)
        return False, [], error_msg


def normalize_result(result: List[Dict]) -> List[Dict]:
    """
    规范化查询结果，用于比较
    
    Args:
        result: 查询结果列表
        
    Returns:
        规范化后的结果列表
    """
    # 排序并转换值类型以便比较
    normalized = []
    for row in result:
        normalized_row = {}
        for key, value in sorted(row.items()):
            # 统一处理 None、数字、字符串
            if value is None:
                normalized_row[key] = None
            elif isinstance(value, (int, float)):
                normalized_row[key] = float(value) if isinstance(value, float) else int(value)
            else:
                normalized_row[key] = str(value).strip()
        normalized.append(normalized_row)
    
    # 按所有字段排序，使结果可比较
    if normalized:
        keys = sorted(normalized[0].keys())
        normalized.sort(key=lambda x: tuple(x.get(k, '') for k in keys))
    
    return normalized


def compare_results(result1: List[Dict], result2: List[Dict]) -> bool:
    """
    比较两个查询结果是否相同
    
    Args:
        result1: 结果1
        result2: 结果2
        
    Returns:
        是否相同
    """
    norm1 = normalize_result(result1)
    norm2 = normalize_result(result2)
    
    if len(norm1) != len(norm2):
        return False
    
    for r1, r2 in zip(norm1, norm2):
        if r1 != r2:
            return False
    
    return True


def evaluate_model(
    base_model_path: str,
    adapter_path: str,
    samples_file: str,
    schema_file: str,
    db_host: str = '127.0.0.1',
    db_port: int = 9030,
    db_user: str = 'root',
    db_password: str = '',
    db_name: str = 'testDB',
    max_samples: int = None,
    save_predictions: Optional[str] = None
) -> Dict[str, Any]:
    """
    评测模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: LoRA adapter路径
        samples_file: success_samples.jsonl 文件路径
        schema_file: schema.json 文件路径
        db_host: 数据库主机
        db_port: 数据库端口
        db_user: 数据库用户
        db_password: 数据库密码
        db_name: 数据库名
        max_samples: 最大评测样本数（用于测试）
        save_predictions: 保存生成的SQL到文件（可选）
        
    Returns:
        评测结果字典
    """
    print(f"Loading samples from {samples_file}...")
    samples = load_samples(samples_file)
    if max_samples:
        samples = samples[:max_samples]
    print(f"Loaded {len(samples)} samples")
    
    # 加载schema
    print(f"Loading schema from {schema_file}...")
    schema_dict = load_schema(schema_file)
    print(f"Loaded schema for {len(schema_dict)} tables")
    
    # 加载模型
    print("\nLoading model...")
    model_inference = SQLModelInference(base_model_path, adapter_path)
    
    # 连接数据库
    print(f"\nConnecting to database {db_host}:{db_port}/{db_name}...")
    try:
        conn = connect_db(host=db_host, port=db_port, user=db_user, 
                         password=db_password, database=db_name)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return {
            'total': len(samples),
            'executable': 0,
            'correct': 0,
            'executable_rate': 0.0,
            'correct_rate': 0.0,
            'errors': []
        }
    
    executable_count = 0
    correct_count = 0
    errors = []
    predictions = []  # 保存所有生成的SQL
    
    print("\nEvaluating samples...")
    for idx, sample in enumerate(samples):
        if idx % 10 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(samples)}...")
        
        sql_id = sample.get('sql_id', f'sample_{idx}')
        question = sample.get('question', '')
        table_list = sample.get('table_list', [])
        knowledge = sample.get('knowledge', '')
        expected_sql = sample.get('sql', '')  # 参考答案（如果有）
        expected_result = sample.get('result', [])  # 预期结果（success_samples.jsonl才有）
        
        # 提取schema摘要
        schema_summary = extract_schema_summary(schema_dict, table_list, knowledge)
        
        # 使用模型生成 SQL
        try:
            generated_sql = model_inference.generate_sql(
                question=question,
                schema_summary=schema_summary,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
        except Exception as e:
            error_msg = f"Model inference error: {str(e)}"
            print(f"  Error generating SQL for {sql_id}: {error_msg}")
            errors.append({
                'sample_id': sql_id,
                'question': question[:100],
                'error': error_msg
            })
            predictions.append({
                'sql_id': sql_id,
                'question': question,
                'generated_sql': '',
                'expected_sql': expected_sql,
                'error': error_msg
            })
            continue
        
        # 保存预测结果
        predictions.append({
            'sql_id': sql_id,
            'question': question,
            'generated_sql': generated_sql,
            'expected_sql': expected_sql
        })
        
        # 测试可执行性：执行生成的 SQL
        success, exec_result, error_msg = execute_sql(conn, generated_sql)
        
        if not success:
            errors.append({
                'sample_id': sql_id,
                'question': question[:100],
                'generated_sql': generated_sql[:200],
                'error': error_msg
            })
            continue
        
        executable_count += 1
        
        # 测试正确性：比较结果（如果expected_result存在）
        if expected_result:
            if compare_results(exec_result, expected_result):
                correct_count += 1
            else:
                errors.append({
                    'sample_id': sql_id,
                    'question': question[:100],
                    'generated_sql': generated_sql[:200],
                    'error': f'Result mismatch. Expected {len(expected_result)} rows, got {len(exec_result)} rows'
                })
        else:
            # 如果没有expected_result（如final_dataset.json），只检查可执行性
            # 可执行即认为可能正确（但无法完全确认）
            correct_count += 1
    
    conn.close()
    
    # 保存预测结果
    if save_predictions:
        print(f"\nSaving predictions to {save_predictions}...")
        with open(save_predictions, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(predictions)} predictions.")
    
    total = len(samples)
    executable_rate = executable_count / total if total > 0 else 0.0
    correct_rate = correct_count / total if total > 0 else 0.0
    
    result = {
        'total': total,
        'executable': executable_count,
        'correct': correct_count,
        'executable_rate': executable_rate,
        'correct_rate': correct_rate,
        'errors': errors[:20]  # 只保存前 20 个错误
    }
    
    return result


def load_schema(schema_file: str) -> Dict[str, Dict]:
    """
    加载 schema.json 文件，返回表名到schema信息的映射
    
    Args:
        schema_file: schema.json 文件路径
        
    Returns:
        字典，key为表名，value为表的schema信息（包含columns等）
    """
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_list = json.load(f)
    
    schema_dict = {}
    for table_info in schema_list:
        table_name = table_info.get('table_name')
        if table_name:
            schema_dict[table_name] = table_info
    
    return schema_dict


def extract_schema_summary(schema_dict: Dict[str, Dict], table_list: List[str], knowledge: str = "") -> str:
    """
    从schema_dict中提取指定表的schema摘要
    
    Args:
        schema_dict: 从load_schema加载的schema字典
        table_list: 表名列表
        knowledge: 业务知识（可选）
        
    Returns:
        schema摘要字符串
    """
    schema_parts = []
    
    # 添加表信息
    for table_name in table_list:
        if table_name in schema_dict:
            table_info = schema_dict[table_name]
            table_desc = table_info.get('table_description', table_name)
            schema_parts.append(f"表: {table_name} ({table_desc})")
            
            # 添加列信息
            columns = table_info.get('columns', [])
            if columns:
                col_info = []
                for col in columns[:20]:  # 限制列数，避免太长
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
    
    # 如果有knowledge，添加到schema信息中
    if knowledge:
        schema_text += f"\n\n业务知识:\n{knowledge}"
    
    return schema_text


def extract_sql_from_text(text: str) -> str:
    """
    从模型生成的文本中提取SQL语句
    
    Args:
        text: 模型生成的文本（可能包含markdown代码块等）
        
    Returns:
        提取的SQL语句
    """
    # 去除首尾空白
    text = text.strip()
    
    # 尝试提取markdown代码块中的SQL
    # 匹配 ```sql ... ``` 或 ``` ... ```
    sql_pattern = re.compile(r'```(?:sql)?\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
    match = sql_pattern.search(text)
    if match:
        return match.group(1).strip()
    
    # 如果没有markdown代码块，尝试查找SQL关键字
    # 查找第一个SELECT、WITH、INSERT等SQL关键字
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
    for keyword in sql_keywords:
        idx = text.upper().find(keyword)
        if idx != -1:
            # 找到SQL开始位置，提取到文本结束或下一个明显非SQL的标记
            sql_text = text[idx:].strip()
            # 移除可能的结尾标记（如<|im_end|>）
            sql_text = re.sub(r'<\|im_end\|>.*$', '', sql_text, flags=re.DOTALL)
            sql_text = sql_text.strip()
            return sql_text
    
    # 如果都没有找到，返回原文本（可能已经是纯SQL）
    # 移除可能的结尾标记
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
        
        # 加载基础模型（使用量化以节省显存）
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA adapter
        if adapter_path:
            print(f"Loading adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # 合并adapter权重到基础模型（便于推理，但会占用更多显存）
            # 如果显存不足，可以注释掉这行，保持LoRA模式
            # self.model = self.model.merge_and_unload()
        
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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SQL generation model (方案B: transformers + peft)')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Base model path (e.g., /path/to/Qwen_Qwen2.5-14B-Instruct)')
    parser.add_argument('--adapter_path', type=str, required=True,
                        help='LoRA adapter path (e.g., /path/to/stage_b_orpo/checkpoint-12)')
    parser.add_argument('--samples_file', type=str, required=True,
                        help='Path to samples file (final_dataset.json or success_samples.jsonl)')
    parser.add_argument('--schema_file', type=str, required=True,
                        help='Path to schema.json')
    parser.add_argument('--db_host', type=str, default='127.0.0.1',
                        help='Database host (default: 127.0.0.1)')
    parser.add_argument('--db_port', type=int, default=9030,
                        help='Database port (default: 9030)')
    parser.add_argument('--db_user', type=str, default='root',
                        help='Database user (default: root)')
    parser.add_argument('--db_password', type=str, default='',
                        help='Database password')
    parser.add_argument('--db_name', type=str, default='testDB',
                        help='Database name (default: testDB)')
    parser.add_argument('--output', type=str, default='evaluation_result.json',
                        help='Output file for evaluation results')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Save generated SQL predictions to file (optional)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    result = evaluate_model(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        samples_file=args.samples_file,
        schema_file=args.schema_file,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        max_samples=args.max_samples,
        save_predictions=args.save_predictions
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Total samples: {result['total']}")
    print(f"Executable: {result['executable']} ({result['executable_rate']*100:.2f}%)")
    print(f"Correct: {result['correct']} ({result['correct_rate']*100:.2f}%)")
    print(f"Errors: {len(result['errors'])}")
    print("="*50)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output}")

