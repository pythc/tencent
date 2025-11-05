import json, re
from pathlib import Path
from collections import Counter, defaultdict

# 改成你的结果文件（WSL 路径）
RESULT = Path("/home/zcy28/workspace/track_3_data_intelligence3/final_for_student/result/dataset_exe_result.json")
OUT_FAILED = Path("/home/zcy28/workspace/track_3_data_intelligence3/final_for_student/result/failed_only.json")

def normalize_msg(msg: str) -> str:
    if not msg: return "UnknownError"
    msg = msg.strip()
    # 归一化常见可变部分（行号、位置、IP、端口等）
    msg = re.sub(r'line \d+','line N', msg, flags=re.I)
    msg = re.sub(r'position \d+','position N', msg, flags=re.I)
    msg = re.sub(r'host [\d\.]+','host X.X.X.X', msg, flags=re.I)
    msg = re.sub(r'port \d+','port N', msg, flags=re.I)
    return msg

def main():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    ok, err = 0, 0
    err_map = defaultdict(list)
    failed_items = []
    for item in data:
        status = item.get("status") or ("result" in item and "status" not in item and "success")  # 兼容早期结构
        if status == "success" or isinstance(item.get("result"), list):
            ok += 1
            continue
        err += 1
        msg = normalize_msg(item.get("error_message",""))
        err_map[msg].append(item.get("sql_id"))
        failed_items.append(item)
    # 输出统计
    print(f"成功: {ok}  失败: {err}  通过率: {ok/(ok+err):.2%}")
    top = Counter({k: len(v) for k,v in err_map.items()}).most_common(10)
    print("\nTop错误类型：")
    for k,cnt in top:
        print(f"- {cnt} | {k}")
    # 失败集合另存，便于只重跑失败
    OUT_FAILED.parent.mkdir(parents=True, exist_ok=True)
    OUT_FAILED.write_text(json.dumps(failed_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已导出仅失败集合: {OUT_FAILED}")

if __name__ == "__main__":
    main()