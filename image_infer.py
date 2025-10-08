from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
from typing import Dict, List
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from PIL import Image
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

# Global flag: 1 -> Chinese prompts, 0 -> English prompts
USE_CHINESE = 1


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_model(model_path: str | Path):
    """Load Qwen2.5-Omni model and processor with memory optimizations."""
    # Load with half precision to save memory
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,  # 使用半精度减少显存占用
        device_map="auto",
        low_cpu_mem_usage=True
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path))
    model.eval()  # 设置为评估模式
    return model, processor


def normalize_image_path(path: str) -> str:
    """Replace old Aistudio dataset root with workspace path if present."""
    if "/home/aistudio/mmallm/data" in path:
        return path.replace("/home/aistudio/mmallm/data", "/root/workspace/mmllm/data")
    return path


def build_conv(record: Dict) -> List[Dict]:
    """Construct conversation structure with user *image* only.

    Prompts are selected based on global USE_CHINESE flag.
    """
    if USE_CHINESE:
        sys_prompt = (
            "你是一名专业的答题助手，请严格按照要求回答问题，如果没有特殊要求，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
        )
        user_prompt = "请回答图片中的问题，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
    else:
        sys_prompt = (
            "You are a professional answer assistant. Please strictly follow the requirements. "
            "Unless otherwise specified, directly provide the letter option of the answer only. "
            "Do not include explanations or any other content."
        )
        user_prompt = "Please answer the question in the image. Directly provide the letter option of the answer, no explanation or other content."

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": normalize_image_path(record["image_path"])},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Core processing

def infer_single(model, processor, record: Dict):
    """Infer answer for a single record with memory optimizations."""
    conv = build_conv(record)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    # Load and process image (automatically resized by processor)
    img_path = Path(normalize_image_path(record["image_path"])).expanduser()
    img = Image.open(img_path).convert('RGB')  # 确保RGB格式

    inputs = processor(
        text=text,
        images=[img],
        return_tensors="pt",
        padding=True  # 需要padding单样本
    )
    
    # 提前关闭图像释放资源
    del img
    gc.collect()
    
    # Move inputs to model device with proper dtype
    inputs = inputs.to(model.device).to(model.dtype)

    try:
        with torch.no_grad():
            text_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                use_audio_in_video=False,
                return_audio=False
            )
        
        # 立即释放输入张量
        del inputs
        torch.cuda.empty_cache()
        
        # 解码结果
        raw_answer = processor.batch_decode(
            text_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        # 释放文本生成张量
        del text_ids
        answer = raw_answer.split("\nassistant\n")[-1].strip()
    
    finally:
        # 确保任何情况下都释放资源
        if 'inputs' in locals():
            del inputs
        if 'text_ids' in locals():
            del text_ids
        torch.cuda.empty_cache()
        gc.collect()
    
    return answer


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(
    model,
    processor,
    input_path: Path,
    output_path: Path,
):
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for idx, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            record: Dict = json.loads(line)
            try:
                answer = infer_single(model, processor, record)
                record["llm_answer"] = answer
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{idx}] → {answer}")
            except Exception as e:
                print(f"Error processing record {idx}: {str(e)}")
                record["llm_answer"] = "ERROR"
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            finally:
                # 每次循环后强制执行垃圾回收
                gc.collect()
                torch.cuda.empty_cache()
    print("Done. Results saved to", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-Omni on prompt images generated by prompt_to_image.py"
    )
    parser.add_argument("--input", type=str, required=True, help="Input *_image.jsonl file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl file for answers (default: <input_basename>_llm_results.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/workspace/models/qwen/Qwen2___5-Omni-7B",
        help="Local path to model checkpoints",
    )
    parser.add_argument(
        "--chinese",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 use English prompts, 1 use Chinese prompts",
    )

    args = parser.parse_args()

    # Set global language flag
    global USE_CHINESE
    USE_CHINESE = args.chinese

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_llm_result.jsonl")
    )

    # Print memory info for debugging
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB / {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    model, processor = load_model(args.model_path)
    process_file(model, processor, input_path, output_path)


if __name__ == "__main__":
    main()
