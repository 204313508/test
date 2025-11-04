"""llm_evaluate.py
使用兼容 OpenAI 接口的大模型对考生答案进行判分。
判分规则：只要最终答案与标准答案完全一致，则输出 1，否则输出 0。
不考虑解题过程，只关心最终答案。
脚本仅输出 1 或 0，不输出任何额外内容。

快速测试方法：
    python llm_evaluate.py
在测试前请将 `API_KEY` 与 `API_BASE` 替换为您的实际值。
"""
from __future__ import annotations

import os
import sys
from typing import Literal

from openai import OpenAI
from tqdm import tqdm

# --------------------------- Prompt Template ---------------------------------
SYSTEM_PROMPT = (
    "你是一名客观的阅卷助手。"
    "任务：判断考生的【最终答案】是否与【标准答案】完全一致。"
    "忽略推理或解题过程的正确性，仅比较最终答案。"
    "若且仅若最终答案与标准答案一致，请回复 1；否则回复 0。"
    "只能回复单个字符 1 或 0，禁止输出任何其他内容。"
)

USER_PROMPT_TEMPLATE = (
    "【题目】\n{question}\n\n"
    "【标准答案】\n{standard}\n\n"
    "【考生作答】（可能包含推理过程）\n{candidate}\n\n"
    "请根据上述信息判断考生的【最终答案】是否与【标准答案】一致。"
    "若一致输出 1，否则输出 0。"
)

# --------------------------- Core Logic --------------------------------------

def evaluate_answer(
    *,
    api_key: str,
    api_base: str,
    question: str,
    standard_answer: str,
    candidate_answer: str,
    model: str = "Qwen/Qwen3-8B",
    temperature: float = 0.0,
) -> Literal[0, 1]:
    """若 *candidate_answer* 的最终答案与 *standard_answer* 完全一致则返回 1，否则返回 0。"""

    if not api_key:
        raise ValueError("API key must be provided.")

    client = OpenAI(api_key=api_key, base_url=api_base)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        standard=standard_answer.strip(),
        candidate=candidate_answer.strip(),
    )

    # 调用 LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=8,
    )

    content = response.choices[0].message.content.strip()
    # print(content)
    if content not in {"0", "1"}:
        raise RuntimeError(f"Unexpected response from model: {content!r}")
    return 1 if content == "1" else 0

# --------------------------- CLI & Batch Evaluation -----------------------

import argparse
import json
from pathlib import Path
import string

def _normalize_answer(ans: str) -> str:
    """Remove whitespace and punctuation for length judgement and normalize case."""
    # Remove whitespace and common punctuation
    tbl = str.maketrans('', '', string.whitespace + string.punctuation)
    return  ans.translate(tbl).lower()

def batch_evaluate(
    *,
    input_path: Path,
    output_path: Path | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str = "Qwen/Qwen3-8B",
) -> float:
    """Evaluate a JSONL dataset and return accuracy."""

    if output_path is None:
        output_path = input_path.with_suffix(".scored.jsonl")

    # ----- Resume support (断点续传) -----
    if output_path.exists():
        # 已存在输出文件，读取已评估的行数和正确数以便跳过
        processed_records = []
        with output_path.open("r", encoding="utf-8") as _resume_fin:
            for _l in _resume_fin:
                if _l.strip():
                    try:
                        _rec = json.loads(_l)
                        processed_records.append(_rec)
                    except Exception:
                        # 若读取失败，则放弃断点续传，重新评估
                        processed_records = []
                        break
        correct = sum(r.get("score", 0) for r in processed_records)
        total = len(processed_records)
        _fout_mode = "a"  # 追加写入
    else:
        correct = 0
        total = 0
        _fout_mode = "w"

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        _fout_mode, encoding="utf-8"
    ) as fout:
        for idx, line in enumerate(tqdm(fin, desc="Evaluating", unit="sample")):
            if idx < total:
                # 已经评估过的记录，直接跳过
                continue
            if not line.strip():
                continue
            record = json.loads(line)
            question = record.get("question", "")
            standard = str(record.get("label", "")).strip()
            candidate = str(record.get("llm_answer", "")).strip()

            # Simple direct comparison after normalization if single char
            norm_candidate = _normalize_answer(candidate)
            if len(norm_candidate) == 1:
                score = 1 if norm_candidate.lower() == standard.lower() else 0
            else:
                print("llm_answer:",norm_candidate)
                print("standard:",standard)
                # Use LLM evaluation
                if api_key is None:
                    raise ValueError(
                        "api_key must be provided for non-trivial answer evaluation"
                    )
                if api_base is None:
                    raise ValueError(
                        "api_base must be provided for non-trivial answer evaluation"
                    )
                score = evaluate_answer(
                    api_key=api_key,
                    api_base=api_base,
                    question=question,
                    standard_answer=standard,
                    candidate_answer=candidate,
                    model=model,
                )

            record["score"] = score
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            correct += score
            total += 1

    accuracy = correct / total if total else 0.0
    return accuracy

# --------------------------- CLI / 快速测试 --------------------------------

if __name__ == "__main__":
    DEFAULT_API_KEY = "sk"
    DEFAULT_API_BASE = "https://api.siliconflow.cn"

    parser = argparse.ArgumentParser(description="Evaluate LLM answers in a JSONL file")
    parser.add_argument("--input", help="Path to input JSONL file")
    parser.add_argument(
        "--output",
        help="Path to output JSONL file with added 'score' field. Default: <input>.scored.jsonl",
    )
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument(
        "--api_base", default=DEFAULT_API_BASE, help="API Base URL"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B", help="Model name for LLM evaluation"
    )
    args = parser.parse_args()

    accuracy = batch_evaluate(
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
    )

    # 按用户要求仅输出准确率数值
    print(accuracy, end="")
