"""Batch-generate SQL with rag_pipeline and execute via sql_exe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rag_pipeline import (
    CorpusConfig,
    SQLRAGPipeline,
    build_llm,
    build_retriever,
)
from sql_exe import execute_sql_with_pymysql
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def load_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    # assume json list
    return json.loads(path.read_text(encoding="utf-8"))


def save_predictions(preds: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_execution(result_path: Path) -> None:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    ok = sum(1 for item in data if item.get("status") == "success")
    err = len(data) - ok
    rate = ok / len(data) * 100 if data else 0.0
    print(f"\nExecution summary: 成功 {ok} 条, 失败 {err} 条, 通过率 {rate:.2f}%")


def load_few_shot_samples(path: Optional[Path]) -> Optional[BM25Retriever]:
    if path is None or not path.exists():
        return None
    docs: List[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = (record.get("question") or "").strip()
            sql = (record.get("sql") or "").strip()
            if not question or not sql:
                continue
            docs.append(Document(page_content=question, metadata={"sql": sql}))
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 8
    return retriever


def build_few_shot_block(question: str, retriever: Optional[BM25Retriever], k: int) -> str:
    if retriever is None or k <= 0:
        return ""
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        docs = retriever.invoke(question)  # type: ignore[attr-defined]
    if not docs:
        return ""
    blocks: List[str] = []
    for doc in docs[:k]:
        sql = doc.metadata.get("sql")
        if not sql:
            continue
        blocks.append(
            f"示例问题:\n{doc.page_content}\n示例SQL:\n```sql\n{sql.strip()}\n```"
        )
    return "\n\n".join(blocks)


INVALID_PATTERNS = [
    " filter ",
    " percentile_",
    "within group",
    " ilike ",
]


def validate_sql_output(sql_text: Optional[str]) -> Optional[str]:
    if not sql_text:
        return None
    sql = sql_text.strip()
    lowered = sql.lower()
    if not lowered.startswith(("select", "with")):
        return None
    for pattern in INVALID_PATTERNS:
        if pattern in lowered:
            return None
    if sql.count(";") > 1:
        return None
    return sql


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate SQL via RAG and execute")
    parser.add_argument("--input", type=Path,
                        default=Path("/home/zcy28/workspace/track_3_data_intelligence3/final_for_student/data/success_samples.jsonl"),
                        help="输入问题文件，支持 jsonl 或 json list，需包含 sql_id/question")
    parser.add_argument("--pred-out", type=Path,
                        default=Path("/home/zcy28/workspace/track_3_data_intelligence3/final_for_student/result/rag_predictions.json"),
                        help="生成 SQL 的输出路径")
    parser.add_argument("--exec-out", type=Path,
                        default=Path("/home/zcy28/workspace/track_3_data_intelligence3/final_for_student/result/rag_execution_result.json"),
                        help="执行结果输出路径")
    parser.add_argument("--llm", choices=["fake", "hf"], default="hf")
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/zcy28/workspace/tencent_cpt/tencent/models/codellama-7b-instruct"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="限制待生成的题目数量，0 表示全部")
    parser.add_argument("--few-shot-file", type=Path,
                        default=Path("/home/zcy28/workspace/track_3_data_intelligence3/final_for_student/data/success_samples.jsonl"),
                        help="few-shot 示例文件（jsonl），包含 question/sql 字段")
    parser.add_argument("--few-shot-k", type=int, default=2, help="每个问题使用的示例数量")
    args = parser.parse_args()

    items = load_items(args.input)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        print("No items to process.")
        return

    print(f"Loaded {len(items)} items from {args.input}")

    corpus_cfg = CorpusConfig(top_k_documents=args.top_k)
    retriever = build_retriever(corpus_cfg)
    llm = build_llm(args.llm, str(args.model_path), max_new_tokens=args.max_new_tokens)
    pipeline = SQLRAGPipeline(retriever=retriever, llm=llm, top_k=args.top_k)
    few_shot_retriever = load_few_shot_samples(args.few_shot_file)

    predictions: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, 1):
        sql_id = item.get("sql_id") or f"item_{idx}"
        question = item.get("question") or ""
        table_list = item.get("table_list") or []
        knowledge = (item.get("knowledge") or "").strip()
        augmented_question = question
        if table_list:
            augmented_question += "\n\n相关表: " + ", ".join(table_list)
        if knowledge:
            augmented_question += f"\n\n业务口径提示:\n{knowledge}"
        few_shot_block = build_few_shot_block(question, few_shot_retriever, args.few_shot_k)
        print(f"[{idx}/{len(items)}] Generating SQL for {sql_id}...")
        try:
            raw_sql = pipeline.generate_sql(augmented_question, few_shots=few_shot_block)
            sql_text = validate_sql_output(raw_sql)
            if raw_sql and not sql_text:
                print(f"{sql_id} 生成 SQL 未通过校验，已丢弃。")
        except Exception as exc:  # capture generation failures
            print(f"Generation failed for {sql_id}: {exc}")
            sql_text = None
        predictions.append({
            "sql_id": sql_id,
            "question": question,
            "sql": sql_text,
        })

    save_predictions(predictions, args.pred_out)
    print(f"Saved predictions to {args.pred_out}")

    # Prepare execution input by filtering out empty SQL (executor会报错，我们事先处理)
    exec_items = [p for p in predictions if p.get("sql")]
    miss = len(predictions) - len(exec_items)
    if miss:
        print(f"跳过 {miss} 条没有生成 SQL 的题目，执行阶段将只处理 {len(exec_items)} 条。")
    exec_input_path = args.pred_out  # reuse same file
    if miss:
        temp_path = args.pred_out.with_suffix(".filtered.json")
        save_predictions(exec_items, temp_path)
        exec_input_path = temp_path

    sql_executor = execute_sql_with_pymysql()
    db_configuration = {
        "host": "127.0.0.1",
        "user": "root",
        "password": "",
        "db": "testDB",
        "port": 9030,
    }
    sql_executor.execute_sql_with_pymysql(str(exec_input_path), str(args.exec_out), db_config=db_configuration)

    if miss:
        exec_input_path.unlink(missing_ok=True)

    summarize_execution(args.exec_out)


if __name__ == "__main__":
    main()
