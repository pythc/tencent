"""SQL RAG pipeline skeleton built on LangChain.

This script illustrates how to wire table schema/knowledge documents into a
retrieval-augmented prompt builder. It intentionally keeps the model layer
pluggable so you can swap in OpenAI, local LM Studio, etc.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

try:  # Optional HuggingFace support
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline as hf_pipeline,
    )
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    ChatHuggingFace = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    hf_pipeline = None  # type: ignore
    torch = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO_ROOT / "data" / "schema.json"
DEFAULT_KNOWLEDGE = REPO_ROOT / "data" / "common_knowledge.md"
DEFAULT_DATASET = REPO_ROOT / "data" / "final_dataset.json"


@dataclass
class CorpusConfig:
    schema_path: Path = DEFAULT_SCHEMA
    knowledge_path: Path = DEFAULT_KNOWLEDGE
    dataset_path: Path = DEFAULT_DATASET
    top_k_documents: int = 6


def load_schema_docs(schema_path: Path) -> List[Document]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for table in schema:
        columns_description = "\n".join(
            f"- {col['col']} ({col['type']}): {col.get('description', '').strip()}"
            for col in table.get("columns", [])
        )
        content = (
            f"Table: {table['table_name']}\n"
            f"Description: {table.get('table_description', '').strip()}\n"
            f"Columns:\n{columns_description}"
        )
        docs.append(Document(page_content=content, metadata={"source": "schema", "table": table["table_name"]}))
    return docs


def load_knowledge_docs(knowledge_path: Path) -> List[Document]:
    text = knowledge_path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": "knowledge"})]


def load_dataset_knowledge(dataset_path: Path) -> List[Document]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for item in dataset:
        knowledge = (item.get("knowledge") or "").strip()
        if not knowledge:
            continue
        content = (
            f"SQL ID: {item.get('sql_id')}\n"
            f"Question: {item.get('question','').strip()}\n"
            f"Guidance: {knowledge}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={"source": "dataset_knowledge", "sql_id": item.get("sql_id")},
            )
        )
    return docs


class SQLRAGPipeline:
    """Minimal RAG pipeline using BM25 retrieval and LangChain prompts."""

    def __init__(
        self,
        retriever: BM25Retriever,
        llm: BaseChatModel,
        top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.base_example = (
            "示例问题:\n统计2024年1月某日在线用户数\n\n"
            "示例上下文:\nTable: dws_mgamejp_login_user_activity_di (字段略)\n\n"
            "示例回答:\n```sql\n"
            "SELECT dtstatdate, COUNT(DISTINCT suserid) AS user_cnt\n"
            "FROM dws_mgamejp_login_user_activity_di\n"
            "WHERE dtstatdate = '20240101'\n"
            "GROUP BY dtstatdate;\n"
            "```"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是专业的 SQL 生成助手，请严格遵循以下要求：\n"
                    "1. 仅输出一段可执行的 StarRocks/MySQL SQL，禁止解释或附加文字。\n"
                    "2. 输出必须放在 ```sql ... ``` 代码块中，且只包含一个查询。\n"
                    "3. 不要使用 PostgreSQL/ANSI 扩展（例如 FILTER、WITHIN GROUP、ILIKE、正则捕获等）。\n"
                    "4. 聚合条件请使用 CASE WHEN，而不是 FILTER。\n"
                    "5. 如需限制结果，使用标准 MySQL 语法；若信息不足请输出 `SELECT 1 WHERE 0;`。\n"
                    "{few_shots}",
                ),
                (
                    "human",
                    "问题:\n{question}\n\n相关上下文:\n{context}\n\n请生成符合业务口径的 SQL。",
                ),
            ]
        )
        self.output_parser = StrOutputParser()

    def build_context(self, question: str) -> str:
        if hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(question)  # older API
        else:
            docs = self.retriever.invoke(question)  # langchain 1.x retriever interface
        if self.top_k > 0:
            docs = docs[: self.top_k]
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(
        self,
        question: str,
        *,
        config: Optional[RunnableConfig] = None,
        few_shots: Optional[str] = None,
    ) -> ChatGeneration:
        context = self.build_context(question)
        chain = self.prompt | self.llm
        few_shot_text = self.base_example
        if few_shots:
            few_shot_text = f"{self.base_example}\n\n{few_shots}"
        return chain.invoke(
            {"question": question, "context": context, "few_shots": few_shot_text},
            config=config,
        )

    def generate_sql(self, question: str, *, few_shots: Optional[str] = None) -> str:
        generation = self.invoke(question, few_shots=few_shots)
        if hasattr(generation, "content"):
            content = generation.content
        elif hasattr(generation, "message"):
            content = generation.message.content
        else:
            content = str(generation)

        text = content.strip() if isinstance(content, str) else str(content)

        # 移除部分模型自动加的指令前缀
        if text.startswith("<s>[INST]") and "[/INST]" in text:
            text = text.split("[/INST]", 1)[1].strip()

        # 尝试从 ```sql ... ``` 中提取
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("sql"):
                    candidate = part[3:].strip()
                else:
                    candidate = part
                if candidate and "select" in candidate.lower():
                    return self._cleanup_sql(candidate)

        # 退一步：从最后一个 SELECT/ WITH 开始截取
        lowered = text.lower()
        for keyword in ["select", "with"]:
            idx = lowered.rfind(keyword)
            if idx != -1:
                return self._cleanup_sql(text[idx:].strip())

        return text

    @staticmethod
    def _cleanup_sql(sql_text: str) -> str:
        sql = sql_text.strip()
        if sql.endswith("```"):
            sql = sql[:-3].rstrip()
        # 截断到最后一个分号后的内容
        if ";" in sql:
            sql = sql.split(";")[0].strip() + ";"
        # 保证只保留首个 SELECT/WITH 之后的内容
        lowered = sql.lower()
        for keyword in ["select", "with"]:
            idx = lowered.find(keyword)
            if idx != -1:
                sql = sql[idx:].strip()
                break
        if not sql.endswith(";") and "select" in sql.lower():
            sql = sql.rstrip() + ";"
        return sql


def build_retriever(config: CorpusConfig) -> BM25Retriever:
    docs = []
    docs.extend(load_schema_docs(config.schema_path))
    docs.extend(load_knowledge_docs(config.knowledge_path))
    docs.extend(load_dataset_knowledge(config.dataset_path))
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = config.top_k_documents
    return retriever


def build_llm(llm_choice: str, model_path: Optional[str], max_new_tokens: int = 512) -> BaseChatModel:
    if llm_choice == "fake":
        preset_sql = "SELECT * FROM dws_jordass_mode_roundrecord_di WHERE dtstatdate = '20250101' LIMIT 10;"
        return FakeListChatModel(responses=[preset_sql])

    if llm_choice == "hf":
        if ChatHuggingFace is None or AutoTokenizer is None or AutoModelForCausalLM is None or hf_pipeline is None or torch is None:
            raise RuntimeError(
                "未检测到 transformers 相关依赖，请先执行 `pip install transformers accelerate sentencepiece` 并确保安装 torch。"
            )
        if not model_path:
            raise ValueError("使用 HuggingFace 模型时必须指定 --model-path 或设置 SQL_RAG_MODEL_PATH")

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
        quant_config = None
        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.float16
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map=device_map,
            local_files_only=True,
        )
        generation_pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        llm_pipeline = HuggingFacePipeline(pipeline=generation_pipeline)
        return ChatHuggingFace(llm=llm_pipeline)

    raise ValueError(f"不支持的 llm 选项: {llm_choice}")


def run_cli(question: str, top_k: int, llm_choice: str, model_path: Optional[str], max_new_tokens: int) -> None:
    corpus_cfg = CorpusConfig(top_k_documents=top_k)
    retriever = build_retriever(corpus_cfg)
    resolved_model_path = model_path or os.getenv("SQL_RAG_MODEL_PATH")
    llm = build_llm(llm_choice, resolved_model_path, max_new_tokens=max_new_tokens)
    pipeline = SQLRAGPipeline(retriever=retriever, llm=llm, top_k=top_k)
    sql = pipeline.generate_sql(question)
    print("\n--- Retrieved Context Preview ---")
    print(pipeline.build_context(question))
    print("\n--- Generated SQL ---")
    print(sql)
    if llm_choice == "fake":
        print("\n(当前仍使用 FakeListChatModel 示例，请替换为实际模型以获得真实效果。)")


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL RAG pipeline skeleton")
    parser.add_argument("question", help="自然语言问题")
    parser.add_argument("--top-k", type=int, default=4, help="返回的检索文档数量")
    parser.add_argument("--llm", choices=["fake", "hf"], default="fake", help="选择底层 LLM，默认使用示例模型")
    parser.add_argument("--model-path", type=str, help="当 --llm=hf 时指定 HuggingFace 模型路径或仓库名，可用环境变量 SQL_RAG_MODEL_PATH 替代")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成 SQL 的最大 Token 数（仅对 HuggingFace 模型生效）")
    args = parser.parse_args()
    run_cli(
        question=args.question,
        top_k=args.top_k,
        llm_choice=args.llm,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
