#!/usr/bin/env python3
"""
StarRocks SQL 适配脚本
将原始 SQL 转换为 StarRocks 兼容的 SQL
"""

import re
from typing import Optional


def adapt_sql_to_starrocks(sql: str) -> str:
    """
    将 SQL 转换为 StarRocks 兼容格式
    
    Args:
        sql: 原始 SQL 字符串
        
    Returns:
        适配后的 SQL 字符串
    """
    if not sql or not sql.strip():
        return sql
    
    # 规范化空白字符
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.strip()
    
    # 1. 移除 FULL OUTER JOIN（StarRocks 不支持）
    # 注意：这是一个简化处理，复杂情况可能需要手动改写
    if 'FULL OUTER JOIN' in sql.upper():
        # 这里只是警告，实际转换需要在复杂情况下手动处理
        print(f"Warning: Found FULL OUTER JOIN in SQL, may need manual conversion: {sql[:100]}...")
    
    # 2. 确保日期字面量加引号（简化处理：识别常见的日期模式）
    # 匹配 YYYYMMDD 格式的裸数字日期（在特定上下文中）
    def add_quotes_to_dates(match):
        date_str = match.group(0)
        # 检查是否已经在引号内
        start_pos = match.start()
        context_before = sql[max(0, start_pos-20):start_pos]
        # 简单的启发式：如果前后有运算符或括号，可能是日期
        if not ('"' in context_before[-10:] or "'" in context_before[-10:]):
            # 检查是否是 8 位数字，可能是日期
            if len(date_str) == 8 and date_str.isdigit():
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return f"'{date_str}'"
        return date_str
    
    # 这个替换要小心，避免误判
    # sql = re.sub(r'\b\d{8}\b', add_quotes_to_dates, sql)
    
    # 3. 统一空值处理：IFNULL -> COALESCE
    sql = re.sub(r'\bIFNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    
    # 4. 字符串函数替换（如果需要）
    # POSITION -> LOCATE（但注意参数顺序可能不同）
    # CHARINDEX -> LOCATE（SQL Server）
    # 这些替换要小心，因为参数顺序可能不同
    
    # 5. 统一引号：将双引号字符串转为单引号（仅在字符串字面量中）
    # 注意：不要转换表名/列名的引号，只转换字符串值
    def fix_string_quotes(match):
        content = match.group(1)
        # 如果内容看起来像字符串值（不是标识符），用单引号
        # 简单的启发式：包含常见字符串字符
        return f"'{content}'"
    
    # 小心替换双引号字符串为单引号（仅在 WHERE/VALUES 等值位置）
    # 这个比较复杂，暂时跳过自动转换，在数据准备时手动处理
    
    # 6. 移除表名/列名的反引号（可选，根据需求）
    # sql = re.sub(r'`(\w+)`', r'\1', sql)  # 移除反引号
    
    # 7. 规范化大小写（可选）
    # SQL 关键字统一大写（可选，StarRocks 不区分大小写）
    # sql = normalize_keywords(sql)
    
    return sql.strip()


def normalize_keywords(sql: str) -> str:
    """
    将 SQL 关键字统一为大写（可选）
    """
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
        'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL OUTER JOIN',
        'UNION', 'UNION ALL', 'EXCEPT', 'INTERSECT',
        'WITH', 'AS', 'ON', 'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'IS', 'NULL',
        'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT',
        'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'CAST', 'COALESCE', 'SUBSTR', 'LOCATE', 'LOWER', 'UPPER',
        'DATE_FORMAT', 'STR_TO_DATE', 'DATE_ADD', 'DATE_SUB',
        'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'OVER',
        'PARTITION BY', 'LIMIT', 'OFFSET'
    ]
    
    # 注意：这个函数可能过于激进，破坏已存在的引号字符串
    # 实际使用时需要更智能的处理
    pattern = r'\b(' + '|'.join(re.escape(kw.lower()) for kw in keywords) + r')\b'
    
    def replace_keyword(match):
        return match.group(0).upper()
    
    # 只在非字符串位置替换（简化处理）
    # 实际应该解析 SQL 避免替换字符串内的关键字
    sql_upper = re.sub(pattern, replace_keyword, sql, flags=re.IGNORECASE)
    
    return sql_upper


if __name__ == '__main__':
    # 测试示例
    test_sqls = [
        "SELECT * FROM table WHERE date_col > 20250724",
        "SELECT IFNULL(col, 0) FROM table",
        "SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id",
        "SELECT POSITION('test' IN col) FROM table",
    ]
    
    for sql in test_sqls:
        adapted = adapt_sql_to_starrocks(sql)
        print(f"Original: {sql}")
        print(f"Adapted:  {adapted}")
        print()

