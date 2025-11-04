# megrez_audio_infer.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import gc
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from modelscope import AutoModelForCausalLM


# Global flag to choose prompt language (1: Chinese, 0: English)
USE_CHINESE = 1


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_model(model_path: str | Path):
    """Load Megrez-3B-Omni model."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    return model


def fix_audio_path(path: str | Path) -> str:
    """Ensure audio path points to the correct location."""
    path_str = str(path)
    if "/home/aistudio/mmallm/data" in path_str:
        path_str = path_str.replace(
            "/home/aistudio/mmallm/data",
            "/root/workspace/mmllm/data",
            1
        )
    # Ensure the file exists
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Audio file not found: {path_str}")
    return path_str


def build_conv(record: Dict) -> List[Dict]:
    """Construct conversation for Megrez: system + user (audio + text)."""
    if USE_CHINESE:
        sys_prompt = (
            "你是一名专业的答题助手，请严格按照要求回答问题，如果没有特殊要求，"
            "请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
        )
        user_prompt = "请回答音频对应的问题，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
    else:
        sys_prompt = (
            "You are a professional answer assistant. Please strictly follow the requirements. "
            "Unless otherwise specified, directly provide the letter option of the answer only. "
            "Do not include explanations or any other content."
        )
        user_prompt = "Please answer the question from the audio. Directly provide the letter option of the answer, no explanation or other content."

    return [
        {"role": "system", "content": {"text": sys_prompt}},
        {
            "role": "user",
            "content": {
                "audio": fix_audio_path(record["audio_path"]),
                "text": record.get("question", user_prompt),
            },
        },
    ]


# ---------------------------------------------------------------------------
# Core processing (single sample)
# ---------------------------------------------------------------------------

def infer_single(model, record: Dict) -> str:
    """Infer answer for a single record using Megrez's chat interface."""
    messages = build_conv(record)
    
    with torch.no_grad():
        response = model.chat(
            messages,
            sampling=True,
            max_new_tokens=4,
            temperature=1,
        )
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    answer = response.strip()
    answer = answer.replace("<|turn_end|>", "").strip()
    return answer


# ---------------------------------------------------------------------------
# File processing (sequential, with resume support)
# ---------------------------------------------------------------------------

def process_file(
    model,
    input_path: Path,
    output_path: Path,
):
    """Process input JSONL file and save results with resume support."""
    total_time = 0
    count = 0
    
    # Count already processed lines
    processed_lines = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_lines += 1

    with input_path.open("r", encoding="utf-8") as fin:
        with output_path.open("a", encoding="utf-8") as fout:
            idx = 0
            for line in fin:
                if not line.strip():
                    continue
                idx += 1
                if idx <= processed_lines:
                    continue  # Skip already processed

                record: Dict = json.loads(line)
                
                start_time = time.time()
                answer = infer_single(model, record)
                elapsed = time.time() - start_time
                
                total_time += elapsed
                count += 1
                
                # try:
                #     answer = infer_single(model, record)
                # except Exception as e:
                #     print(f"❌ Error on line {idx}: {e}")
                #     answer = "[ERROR]"

                record["llm_answer"] = answer
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                if idx % 10 == 0:
                    fout.flush()
                    print(f"[{idx}] records processed (resumed) (Time: {elapsed:.3f}s)")
                else:
                    print(f"[{idx}] Processed (Time: {elapsed:.3f)s)")

    if count > 0:
        avg_time = total_time / count
        print(f"\n✅ 平均每条推理时间: {avg_time:.3f} 秒 (共 {count} 条)")
    else:
        print("⚠️ 未处理任何样本。")
    
    print("✅ Done. Results saved to", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Megrez-3B-Omni on audio QA dataset (CEval audio version)"
    )
    parser.add_argument("--input", type=str, required=True, help="Input *_audio.jsonl file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl file (default: <input_basename>_llm_result.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path to Megrez-3B-Omni model",
    )
    parser.add_argument(
        "--chinese",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 = English prompts, 1 = Chinese prompts",
    )

    args = parser.parse_args()

    global USE_CHINESE
    USE_CHINESE = args.chinese

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_llm_result.jsonl")
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path)
    process_file(model, input_path, output_path)


if __name__ == "__main__":
    main()