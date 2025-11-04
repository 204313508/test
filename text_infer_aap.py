"""test.py
在mmlu测试集上本地运行Qwen2.5-Omni-3B，并保存预测结果。

此脚本假设模型权重已通过`download_model.py`下载到`./models`目录中
（默认路径为`./models/Qwen/Qwen2.5-Omni-3B`）。

用法::

    python test.py --input data/mmlu/test.jsonl --output result.jsonl \
                   --model_path ./models/Qwen/Qwen2.5-Omni-3B

生成的JSONL行保留原始字段，并附加模型的原始答案`llm_answer`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import time

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

# Global flag to control language of prompts (1: Chinese, 0: English)
USE_CHINESE = 1


def load_model(model_path: str | Path):
    """加载模型及其处理器。

    参数:
    - model_path (str | Path): 模型的本地路径。

    返回:
    - model: 已加载的模型。
    - processor: 与模型相关联的处理器。
    """
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    # Ensure pad_token is defined to prevent generation issues
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure consistency between tokenizer and model generation config
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model = model.eval()
    return model, tokenizer


def build_prompt(record: dict) -> str:
    """构建聊天模板（第一轮），仅向模型介绍题目，不要求直接给出答案。

    The content language is chosen based on global USE_CHINESE.
    第一轮：给模型题目和选项，让其介绍题目背景和考察知识点；
    第二轮：再请求模型仅给出答案字母选项，不包含解释。
    """
    if USE_CHINESE:
        system_text = (
            "你是一名专业的答题助手，请严格按照要求回答问题，如果没有特殊要求，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
        )
        # 第一轮对话内容：仅提供题目信息，让模型介绍背景和考察知识点
        user_text = (
            f"{record['question']}\n"
            f"选项为：A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "请不要回答答案，只需介绍题目背景以及考察的知识点。"
        )
    else:
        system_text = (
            "You are a professional answer assistant. Please strictly follow the requirements. "
            "Unless otherwise specified, directly provide the letter option of the answer only. "
            "Do not include explanations or any other content."
        )
        user_text = (
            f"{record['question']}\n"
            f"Options: A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "Please do NOT provide the answer yet. Simply introduce the background of the question and the knowledge points assessed."
        )

    conversation = [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]
    return conversation


def infer(model, tokenizer, record: dict) -> str:
    """进行推理，返回模型的答案。

    参数:
    - model: 已加载的模型。
    - processor: 处理器。
    - record (dict): 题目记录。

    返回:
    - answer (str): 模型的答案。
    """
    # -------- 第一轮：仅让模型介绍背景 ---------
    conversation = build_prompt(record)
    text_round1 = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs_round1 = tokenizer([text_round1], return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Use greedy decoding to avoid sampling-related CUDA asserts on CUDA kernels
        output_round1 = model.generate(
            **inputs_round1,
            max_new_tokens=512,
            do_sample=False,  # disable sampling
        )

    # 获取第一轮 assistant 内容
    output_round1_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_round1.input_ids, output_round1)]
    assistant_round1 = tokenizer.batch_decode(output_round1_trimmed, skip_special_tokens=True)[0]

    # -------- 第二轮：请求模型直接给出答案 ---------
    if USE_CHINESE:
        second_user_msg = "请直接给出答案的字母选项，不要包含任何解释或其他内容，仅输出一个大写字母。"
    else:
        second_user_msg = "Please provide ONLY the letter option of the answer in uppercase. Do not include any explanations or extra content."

    conversation.append({"role": "assistant", "content": assistant_round1})
    conversation.append({"role": "user", "content": second_user_msg})

    text_round2 = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs_round2 = tokenizer([text_round2], return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Greedy decoding for the final answer
        output_round2 = model.generate(
            **inputs_round2,
            max_new_tokens=16,
            do_sample=False,
        )

    output_round2_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_round2.input_ids, output_round2)]
    raw_answer = tokenizer.batch_decode(output_round2_trimmed, skip_special_tokens=True)[0]

    # 只保留字母部分，去除潜在空格/标点
    for ch in ["A", "B", "C", "D"]:
        if ch in raw_answer:
            return ch
    # 如果未能解析，返回原始文本便于排查
    return raw_answer


def process_file(model, tokenizer, input_path: Path, output_path: Path):
    """处理输入文件，生成并保存预测结果。

    参数:
    - model: 已加载的模型。
    - processor: 处理器。
    - input_path (Path): 输入文件路径。
    - output_path (Path): 输出文件路径。
    """
    total_time = 0
    count = 0
    
    # 断点续传功能：如果输出文件已存在，则统计其中已处理的非空行数，以便跳过输入文件中对应的行。
    processed_count = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing_f:
            processed_count = sum(1 for _ in existing_f if _.strip())
        print(f"Resuming from previous run, skipping {processed_count} already processed lines.")

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx < processed_count:
                continue
            if not line.strip():
                continue
            record: Dict = json.loads(line)
            
            start_time = time.time()
            llm_answer = infer(model, tokenizer, record)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            count += 1
            
            record["llm_answer"] = llm_answer.split("答案：")[-1].split("答案:")[-1].split("Answer:")[-1].split("Answer：")[-1]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"Processed {record.get('question')[:30]} -> {record['llm_answer']} (Time: {elapsed:.3f}s)")
            torch.cuda.empty_cache()
    
    if count > 0:
        avg_time = total_time / count
        print(f"\n✅ 平均每条推理时间: {avg_time:.3f} 秒 (共 {count} 条)")
    else:
        print("⚠️ 未处理任何样本。")


def main():
    """主函数，处理命令行参数并执行模型推理。"""
    parser = argparse.ArgumentParser(description="使用Qwen2.5-Omni-3B运行mmlu测试集")
    parser.add_argument(
        "--input",
        type=str,
        default="data/mmlu/test.jsonl",
        help="mmlu测试集的JSONL文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mmlu_text_llm_aap_3b_result.jsonl",
        help="保存预测结果的路径（JSONL格式）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/pretrainmodel/Qwen2.5-3B-Instruct",
        help="包含模型检查点的本地目录",
    )
    parser.add_argument(
        "--chinese",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 使用英文提示, 1 使用中文提示",
    )
    args = parser.parse_args()

    # Set global language preference
    global USE_CHINESE
    USE_CHINESE = args.chinese

    model, tokenizer = load_model(args.model_path)
    process_file(model, tokenizer, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
