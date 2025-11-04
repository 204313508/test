"""
在CEval测试集上本地运行 Megrez-3B-Omni，并保存预测结果。

用法::

    python test_megrez.py --input data/ceval/test.jsonl --output result.jsonl \
                          --model_path ./models/AI-ModelScope/Megrez-3B-Omni

生成的JSONL行保留原始字段，并附加模型的原始答案 `llm_answer`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from modelscope import AutoModelForCausalLM

import time


# Global flag to control language of prompts (1: Chinese, 0: English)
USE_CHINESE = 1


def load_model(model_path: str | Path):
    """加载 Megrez 模型。"""
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        .eval()
        .cuda()
    )
    return model


def build_prompt(record: dict):
    """构建聊天模板，指示模型仅输出答案字母。"""
    if USE_CHINESE:
        system_text = "你是一名专业的答题助手，请严格按照要求回答问题。如果没有特殊要求，请直接给出答案的字母选项（A/B/C/D），不要包含解释或其他内容。"
        user_text = (
            f"{record['question']}\n"
            f"选项为：A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "请直接给出答案的字母（A/B/C/D），不要写其他内容。"
        )
    else:
        system_text = "You are a professional assistant. Only answer with the letter option (A/B/C/D). Do not include explanations or other content."
        user_text = (
            f"{record['question']}\n"
            f"Options: A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "Please answer with A/B/C/D only."
        )

    messages = [
        {"role": "system", "content": {"text": system_text}},
        {"role": "user", "content": {"text": user_text}},
    ]
    return messages


@torch.inference_mode()
# def infer(model, record: dict) -> str:
#     """对单条记录进行推理，返回模型预测答案。"""
#     messages = build_prompt(record)
#     response = model.chat(
#         messages,
#         sampling=False,
#         max_new_tokens=4,
#         temperature=0,
#     )
#     # 取最后的预测字符串
#     return response.strip()
def infer(model, record: dict) -> str:
    conversation = build_prompt(record)
    response = model.chat(
        conversation,
        sampling=False,
        max_new_tokens=10,
        temperature=0,
    )
    # 去掉特殊 token
    answer = response.strip()
    answer = answer.replace("<|turn_end|>", "").strip()
    return answer



def process_file(model, input_path: Path, output_path: Path):

    total_time = 0
    count = 0
    """逐行处理 JSONL 文件，追加写入预测结果。"""
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            if count >= 10:
                break
            if not line.strip():
                continue
            record: Dict = json.loads(line)

            start_time = time.time()

            llm_answer = infer(model, record)
            elapsed = time.time() - start_time
            total_time += elapsed
            count += 1
            record["llm_answer"] = llm_answer
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            # print("Processed", record.get("question")[:30], "->", llm_answer)
            print(f"Processed {count}/{10}: {record.get('question', '')[:30]} -> {llm_answer} (Time: {elapsed:.3f}s)")
            torch.cuda.empty_cache()
    if count > 0:
        avg_time = total_time / count
        print(f"\n✅ 平均每条推理时间: {avg_time:.3f} 秒 (共 {count} 条)")
    else:
        print("⚠️ 未处理任何样本。")


def main():
    parser = argparse.ArgumentParser(description="使用 Megrez-3B-Omni 运行CEval测试集")
    parser.add_argument("--input", type=str, default="data/ceval/test.jsonl", help="输入JSONL文件路径")
    parser.add_argument("--output", type=str, default="ceval_megrez_result.jsonl", help="保存预测结果的路径")
    parser.add_argument("--model_path", type=str, default="/root/workspace/models/AI-ModelScope/Megrez-3B-Omni", help="模型本地路径")
    parser.add_argument("--chinese", type=int, choices=[0, 1], default=1, help="0 使用英文提示, 1 使用中文提示")
    # parser.add_argument("--max_samples", type=int, default=10, help="最大样本数")
    args = parser.parse_args()

    global USE_CHINESE
    USE_CHINESE = args.chinese

    model = load_model(args.model_path)
    process_file(model, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
