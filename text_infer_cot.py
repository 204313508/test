"""test.py
在CEval测试集上本地运行Qwen2.5-Omni-3B，并保存预测结果。

此脚本假设模型权重已通过`download_model.py`下载到`./models`目录中
（默认路径为`./models/Qwen/Qwen2.5-Omni-3B`）。

用法::

    python test.py --input data/ceval/test.jsonl --output result.jsonl \
                   --model_path ./models/Qwen/Qwen2.5-Omni-3B

生成的JSONL行保留原始字段，并附加模型的原始答案`llm_answer`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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
        str(model_path), torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return model, tokenizer


def build_prompt(record: dict) -> str:
    """构建聊天模板，指示模型仅输出选项 / Build chat template for the model.

    The content language is chosen based on global USE_CHINESE.
    """
    if USE_CHINESE:
        system_text = (
            "你是一名专业的答题助手，请严格按照要求回答问题，如果没有特殊要求，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
        )
        user_text = (
            f"{record['question']}\n"
            f"选项为：A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "让我们一步步思考。请先输出思考过程，然后换两行后输出答案选项，格式为\"思考过程\\n\\n答案：\""
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
            "Let's think step by step. First, provide your reasoning, then add two line breaks and output the final answer option in the format \"Thought\n\nAnswer:\""
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
    conversation = build_prompt(record)
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    # Remove prompt tokens
    generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    raw_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Expect output in "思考过程\n\n答案：X" format, return as-is for now
    return raw_answer


def process_file(model, tokenizer, input_path: Path, output_path: Path):
    """处理输入文件，生成并保存预测结果。

    参数:
    - model: 已加载的模型。
    - processor: 处理器。
    - input_path (Path): 输入文件路径。
    - output_path (Path): 输出文件路径。
    """
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
            llm_answer = infer(model, tokenizer, record)
            record["llm_answer"] = llm_answer.split("答案：")[-1].split("答案:")[-1].split("Answer:")[-1].split("Answer：")[-1]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print("Processed", record.get("question")[:30], "->", llm_answer)
            torch.cuda.empty_cache()


def main():
    """主函数，处理命令行参数并执行模型推理。"""
    parser = argparse.ArgumentParser(description="使用Qwen2.5-Omni-3B运行CEval测试集")
    parser.add_argument(
        "--input",
        type=str,
        default="data/ceval/test.jsonl",
        help="CEval测试集的JSONL文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ceval_text_llm_result.jsonl",
        help="保存预测结果的路径（JSONL格式）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Qwen/Qwen2.5-3B-Instruct",
        help="包含模型检查点的本地目录",
    )
    parser.add_argument(
        "--chinese",
        type=int,
        choices=[0, 1],
        default=1,
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
