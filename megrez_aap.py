"""test_megrez_aap.py
在MMLU测试集上本地运行 Megrez-3B-Omni，并保存预测结果（使用Answer-Aware Prompting, AAP）。

此脚本假设模型权重已下载到指定路径。

用法::

    python test_megrez_aap.py --input data/mmlu/test.jsonl --output result.jsonl \
                              --model_path ./models/Megrez-3B-Omni

生成的JSONL行保留原始字段，并附加模型的答案`llm_answer`（仅A/B/C/D）。
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
USE_CHINESE = 0


def load_model(model_path: str | Path):
    """加载 Megrez-3B-Omni 模型。

    参数:
    - model_path (str | Path): 模型的本地路径。

    返回:
    - model: 已加载的模型（已移至 GPU 并设为 eval 模式）。
    """
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    ).eval().cuda()
    return model


def build_first_round_messages(record: dict) -> list[dict]:
    """构建第一轮对话：仅让模型介绍题目背景和知识点（不回答答案）。"""
    if USE_CHINESE:
        user_text = (
            f"{record['question']}\n"
            f"选项为：A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "请不要回答答案，只需介绍题目背景以及考察的知识点。"
        )
    else:
        user_text = (
            f"{record['question']}\n"
            f"Options: A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "Please do NOT provide the answer yet. Simply introduce the background of the question and the knowledge points assessed."
        )

    return [
        {
            "role": "user",
            "content": {"text": user_text}
        }
    ]


def build_second_round_messages(first_round_messages: list[dict], first_response: str) -> list[dict]:
    """构建第二轮对话：在第一轮基础上追加 assistant 回复和新 user 请求（仅输出答案）。"""
    # Megrez 的 chat 接口要求完整对话历史
    conversation = first_round_messages.copy()
    conversation.append({
        "role": "assistant",
        "content": {"text": first_response}
    })

    if USE_CHINESE:
        second_user_text = "请直接给出答案的字母选项，不要包含任何解释或其他内容，仅输出一个大写字母。"
    else:
        second_user_text = "Please provide ONLY the letter option of the answer in uppercase. Do not include any explanations or extra content."

    conversation.append({
        "role": "user",
        "content": {"text": second_user_text}
    })
    return conversation


def infer(model, record: dict) -> str:
    """执行两阶段 AAP 推理，返回最终答案（A/B/C/D）。"""
    # ===== 第一阶段：获取题目分析 =====
    messages_round1 = build_first_round_messages(record)
    with torch.no_grad():
        response_round1 = model.chat(
            messages_round1,
            sampling=False,
            max_new_tokens=512,
            temperature=0.0,
        )

    # ===== 第二阶段：请求仅输出答案 =====
    messages_round2 = build_second_round_messages(messages_round1, response_round1)
    with torch.no_grad():
        response_round2 = model.chat(
            messages_round2,
            sampling=False,
            max_new_tokens=16,  # 答案很短
            temperature=0.0,
        )

    # 尝试提取 A/B/C/D
    response_clean = response_round2.strip().upper()
    for option in ["A", "B", "C", "D"]:
        if option in response_clean:
            # 确保是独立字母（避免 "AB" 等情况）
            if response_clean == option or (response_clean.startswith(option) and (len(response_clean) == 1 or not response_clean[1].isalpha())):
                return option

    # 若无法解析，返回原始响应（便于调试）
    return response_round2


def process_file(model, input_path: Path, output_path: Path):
    total_time = 0
    count = 0
    """处理输入文件，生成并保存预测结果（支持断点续传）。"""
    processed_count = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            processed_count = sum(1 for line in f if line.strip())
        print(f"Resuming from previous run, skipping {processed_count} already processed lines.")

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if count >= 10:
                break
            if idx < processed_count:
                continue
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
            print(f"Processed {idx}: {record.get('question')[:30]}... -> {llm_answer}")
            torch.cuda.empty_cache()

    if count > 0:
        avg_time = total_time / count
        print(f"\n✅ 平均每条推理时间: {avg_time:.3f} 秒 (共 {count} 条)")
    else:
        print("⚠️ 未处理任何样本。")


def main():
    parser = argparse.ArgumentParser(description="使用 Megrez-3B-Omni 运行 MMLU 测试集（AAP 两阶段推理）")
    parser.add_argument("--input", type=str, default="data/mmlu/test.jsonl", help="MMLU测试集路径")
    parser.add_argument("--output", type=str, default="megrez_aap_result.jsonl", help="输出结果路径")
    parser.add_argument("--model_path", type=str, required=True, help="Megrez-3B-Omni 模型路径")
    parser.add_argument("--chinese", type=int, choices=[0, 1], default=0, help="0: 英文提示, 1: 中文提示")

    args = parser.parse_args()

    global USE_CHINESE
    USE_CHINESE = args.chinese

    print("Loading Megrez-3B-Omni model...")
    model = load_model(args.model_path)
    print("Model loaded.")

    process_file(model, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()