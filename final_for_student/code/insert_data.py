#!/usr/bin/env python3
"""数据插入脚本 - 使用sql_exe.py中的insert_data_with_pymysql函数插入测试数据"""
import sys
from pathlib import Path

# 添加当前目录到路径，以便导入sql_exe模块
sys.path.insert(0, str(Path(__file__).parent))

from sql_exe import execute_sql_with_pymysql

def main():
    """主函数：执行数据插入"""
    # 数据库连接配置
    db_configuration = {
        "host": "127.0.0.1",
        "user": "root",
        "password": "",
        "db": "testDB",
        "port": 9030,
    }
    
    # 文件路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "insert_sql.json"
    output_file = project_root / "result" / "insert_execution_result.json"
    
    print(f"读取插入SQL文件: {input_file}")
    print(f"输出结果文件: {output_file}")
    
    # 创建SQL执行器
    sql_executor = execute_sql_with_pymysql()
    
    # 执行插入操作
    print("\n开始执行数据插入...")
    sql_executor.insert_data_with_pymysql(
        str(input_file),
        str(output_file),
        db_config=db_configuration
    )
    
    # 统计结果
    import json
    if output_file.exists():
        results = json.loads(output_file.read_text(encoding="utf-8"))
        success = sum(1 for r in results if r.get("status") == "success")
        error = len(results) - success
        print(f"\n插入完成！成功: {success} 条, 失败: {error} 条")
        
        if error > 0:
            print("\n失败的插入项:")
            for r in results:
                if r.get("status") != "success":
                    print(f"  - {r.get('sql_id')}: {r.get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    main()



