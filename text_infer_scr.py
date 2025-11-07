"""text_infer_scr.py
在CEval测试集上本地运行Qwen2.5-Omni-3B，使用SCR提示格式，并保存预测结果。

此脚本假设模型权重已通过`download_model.py`下载到`./models`目录中
（默认路径为`./models/Qwen/Qwen2.5-Omni-3B`）。

用法::

    python text_infer_scr.py --input data/ceval/test.jsonl --output result_scr.jsonl \
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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

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
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path))
    return model, processor


def build_initial_prompt(record: dict) -> list:
    """构建SCR格式的初始聊天模板，指示模型进行自我反思和一致性检查。

    参数:
    - record (dict): 包含题目、选项等信息的字典，键包括 question、A、B、C、D。

    返回:
    - conversation (list): 适用于 Qwen Processor 的聊天模板。
    """
    if USE_CHINESE:
        system_text = (
            "你是一名专业的答题助手，请严格按照要求回答问题，如果没有特殊要求，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
        )
        # 构建SCR格式的提示
        user_text = (
            f"[Turn 1] User: 以下问题可能包含字母替换等干扰（例如用"degere"代替"degree"，用"hte"代替"the"）。请先识别并纠正任何错误，然后再解决问题。"
            f"干扰问题示例：\"{record['question']}\"\n"
            f"选项：A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
        )
    else:
        system_text = (
            "You are a professional answer assistant. Please strictly follow the requirements. "
            "Unless otherwise specified, directly provide the letter option of the answer only. "
            "Do not include explanations or any other content."
        )
        # 构建SCR格式的提示
        user_text = (
            f"[Turn 1] User: The following question may contain perturbations such as letter substitutions (e.g., \"degere\" for \"degree\", \"hte\" for \"the\"). "
            f"Please first identify and correct any errors before solving. "
            f"Perturbed question: \"{record['question']}\"\n"
            f"Options: A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
        )

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]
    return conversation


def get_next_turn_prompts() -> tuple:
    """获取后续轮次的提示文本。
    
    返回:
    - tuple: (user_text2, user_text3) 后续两轮用户提示
    """
    if USE_CHINESE:
        user_text2 = "很好。请简要解释相关背景知识，然后解决问题。"
        user_text3 = "请仅提供答案的字母选项（大写），不要包含解释或额外内容。"
    else:
        user_text2 = "Great. Please explain the relevant background briefly and then solve it."
        user_text3 = "Please provide ONLY the letter option of the answer in uppercase. Do not include any explanations or extra content."
    
    return user_text2, user_text3


def generate_response(model, processor, conversation: list, max_new_tokens: int = 256) -> str:
    """生成模型回答。
    
    参数:
    - model: 已加载的模型。
    - processor: 处理器。
    - conversation (list): 对话历史。
    - max_new_tokens (int): 最大生成token数。
    
    返回:
    - response (str): 模型的回答。
    """
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_audio_in_video=False, return_audio=False)
    raw_response = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # 提取最新一轮的回答
    response_parts = raw_response.split("\nassistant\n")
    if len(response_parts) > 1:
        response = response_parts[-1].strip()
    else:
        response = raw_response.strip()
    
    return response


def infer(model, processor, record: dict) -> str:
    """进行多轮推理，返回模型的最终答案。

    参数:
    - model: 已加载的模型。
    - processor: 处理器。
    - record (dict): 题目记录。

    返回:
    - answer (str): 模型的最终答案。
    """
    # 第一轮：构建初始对话并获取模型回答
    conversation = build_initial_prompt(record)
    response1 = generate_response(model, processor, conversation)
    
    # 将第一轮回答添加到对话历史
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response1}],
    })
    
    # 第二轮：添加用户提示并获取模型回答
    user_text2, user_text3 = get_next_turn_prompts()
    conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text2}],
    })
    response2 = generate_response(model, processor, conversation)
    
    # 将第二轮回答添加到对话历史
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response2}],
    })
    
    # 第三轮：添加最终用户提示并获取模型回答
    conversation.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text3}],
    })
    final_response = generate_response(model, processor, conversation, max_new_tokens=4)
    
    # 尝试提取单个字母答案
    import re
    match = re.search(r'\b([A-D])\b', final_response)
    if match:
        answer = match.group(1)
    else:
        answer = final_response.strip()
    
    return answer


def process_file(model, processor, input_path: Path, output_path: Path):
    """处理输入文件，生成并保存预测结果。

    参数:
    - model: 已加载的模型。
    - processor: 处理器。
    - input_path (Path): 输入文件路径。
    - output_path (Path): 输出文件路径。
    """
    total_time = 0
    count = 0
    
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record: Dict = json.loads(line)
            
            start_time = time.time()
            llm_answer = infer(model, processor, record)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            count += 1
            
            record["llm_answer"] = llm_answer
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"Processed {record.get('question')[:30]} -> {llm_answer} (Time: {elapsed:.3f}s)")
            torch.cuda.empty_cache()
    
    if count > 0:
        avg_time = total_time / count
        print(f"\n✅ 平均每条推理时间: {avg_time:.3f} 秒 (共 {count} 条)")
    else:
        print("⚠️ 未处理任何样本。")


def main():
    """主函数，处理命令行参数并执行模型推理。"""
    parser = argparse.ArgumentParser(description="使用Qwen2.5-Omni-3B运行CEval测试集（SCR格式）")
    parser.add_argument(
        "--input",
        type=str,
        default="data/ceval/perturbed/perturbed_ceval_test_20.jsonl",
        help="CEval测试集的JSONL文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ceval_text_llm_result_scr.jsonl",
        help="保存预测结果的路径（JSONL格式）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Qwen/Qwen2.5-Omni-3B",
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

    model, processor = load_model(args.model_path)
    process_file(model, processor, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()