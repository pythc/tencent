# StarRocks SQL 适配规则

本文档定义了 CSpider 数据转换为 StarRocks 兼容 SQL 的最小改写规则。

## 日期/字面量

- **统一日期格式**：全部日期写成字符串 `'YYYYMMDD'` 或 `'YYYY-MM-DD'`（统一使用一种格式）
- **避免裸数字日期**：避免 `20250724` 这样的格式，统一加引号 `'20250724'`
- **时间区间统一含义**：
  - 闭区间按日粒度：`BETWEEN '20250701' AND '20250731'`
  - 带时间精度就用闭开：`>= '2025-07-01 00:00:00' AND < '2025-08-01 00:00:00'`

## 函数/运算

- **空值处理**：统一使用 `COALESCE(x, y)`（`IFNULL` 也可，但统一一种）
- **字符串函数**：
  - 统一使用 `SUBSTR(s, pos, len)`
  - 统一使用 `LOCATE(substr, s)`
  - 避免使用 `POSITION`、`CHARINDEX` 等方言函数
- **大小写**：如需统一可用 `LOWER(s)` / `UPPER(s)`；避免方言函数（如 Postgres 的 `ILIKE`）
- **时间格式化**：
  - `DATE_FORMAT(ts, '%Y-%m-%d')` 可用
  - `STR_TO_DATE(str, '%Y-%m-%d')` 可用
  - 避免不常见时间函数
- **类型转换**：使用 `CAST(x AS <TYPE>)`，或 `CONVERT` 的 MySQL 形态

## 连接/集合

- **禁用 FULL OUTER JOIN**：StarRocks 不支持，改写为：
  - `LEFT JOIN ... UNION ALL ...` + 反向过滤
  - 或用 `LEFT JOIN` / `RIGHT JOIN` 组合
- **优先显式 JOIN + ON**：避免 `NATURAL JOIN`
- **UNION 操作**：使用 `UNION` 或 `UNION ALL`，注意去重语义

## 窗口/聚合

- **窗口函数**：按 ANSI 写法没问题：
  - `ROW_NUMBER() OVER(PARTITION BY ... ORDER BY ...)`
  - `DENSE_RANK()` / `LAG()` / `LEAD()` 等同理
- **条件聚合**：统一写法 `SUM(CASE WHEN cond THEN 1 ELSE 0 END)`

## 标识/风格

- **CTE 优先**：统一使用 `WITH ... AS (...)` 编排复杂查询
- **别名风格**：别名小写蛇形（snake_case）
- **引号统一**：字符串统一使用单引号 `'`
- **标识符**：
  - 表/列名尽量不包反引号
  - 确需包裹时统一使用反引号 `` `col` ``
- **谓词下推友好**：避免函数包住分区列，优先直写常量比较

## 示例转换

### 日期转换示例
```sql
-- 原 SQL (假设)
WHERE date_col > 20250724

-- 转换后
WHERE date_col > '20250724'
```

### JOIN 转换示例
```sql
-- 原 SQL (假设包含 FULL OUTER JOIN)
SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id

-- 转换后
SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id
UNION ALL
SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id WHERE t1.id IS NULL
```

### 空值处理示例
```sql
-- 统一使用 COALESCE
SELECT COALESCE(col, 0) FROM table
```

### 字符串函数示例
```sql
-- 统一使用 SUBSTR 和 LOCATE
SELECT SUBSTR(name, 1, 10) FROM table WHERE LOCATE('test', name) > 0
```

