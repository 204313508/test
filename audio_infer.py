from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchaudio  # type: ignore
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

# Global flag to choose prompt language (1: Chinese, 0: English)
USE_CHINESE = 1


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_model(model_path: str | Path):
    """Load Qwen2.5-Omni model and processor."""
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch.float16, device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path))
    model.disable_talker()
    return model, processor



def build_conv(record: Dict) -> List[Dict]:
    """Construct conversation structure with user *audio* only.

    Prompts switch language based on USE_CHINESE.
    """
    if USE_CHINESE:
        sys_prompt = (
            "你是一名专业的答题助手，请严格按照要求回答问题，如果没有特殊要求，请直接给出答案的字母选项，不要包含解释，不要有其他内容。"
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
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": record["audio_path"]},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Core processing (single sample)
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = 16000):
    """Load audio file and resample to *target_sr* mono tensor."""
    path_str = str(path)
    if "/home/aistudio/mmallm/data" in path_str:
        path_str = path_str.replace(
            "/home/aistudio/mmallm/data",
            "/root/workspace/mmllm/data",
            1  # Only replace the first occurrence
        )
    
    waveform, sr = torchaudio.load(path_str)  # (channels, time)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Qwen expects float32
    return waveform.squeeze(0).cpu().numpy()


def infer_single(model, processor, record: Dict):
    """Infer answer for a single record (sequential processing)."""
    conv = build_conv(record)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    audio_tensor = load_audio(Path(record["audio_path"]).expanduser())

    inputs = processor(text=text, audio=[audio_tensor], return_tensors="pt", use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids = model.generate(
            **inputs, max_new_tokens=4, use_audio_in_video=True, return_audio=False
        )
    raw_answer = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    answer = raw_answer.split("\nassistant\n")[-1].strip()

    # Clear CUDA cache and collect garbage to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return answer


# ---------------------------------------------------------------------------
# Core processing (batch)
# ---------------------------------------------------------------------------

def batch(iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def infer_audios(
    model,
    processor,
    records: List[Dict],
):
    """Run inference for a list of *records* and return answers."""
    conversations = [build_conv(r) for r in records]

    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )

    audio_tensors = [load_audio(Path(r["audio_path"]).expanduser()) for r in records]

    inputs = processor(
        text=text,
        audio=audio_tensors,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            use_audio_in_video=True,
            return_audio=False,
        )
    answers = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # Clear CUDA cache and collect garbage to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return [a.split("\nassistant\n")[-1].strip() for a in answers]


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(
    model,
    processor,
    input_path: Path,
    output_path: Path,
):
    """Process *input_path* jsonl file sequentially and save answers to *output_path* with checkpoint resume (断点续传)."""
    # Check how many lines already processed (non-empty lines)
    processed_lines = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as fout:
            for line in fout:
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
                    continue  # Skip already processed lines
                record: Dict = json.loads(line)
                # run inference for this single record
                answer = infer_single(model, processor, record)
                record["llm_answer"] = answer
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                # flush periodically and log progress
                if idx % 10 == 0:
                    fout.flush()
                    print(f"[{idx}] records processed (resumed)")
    print("Done. Results saved to", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-Omni on prompt audios generated by prompt_to_audio.py"
    )
    parser.add_argument("--input", type=str, required=True, help="Input *_audio.jsonl file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl file for answers (default: <input_basename>_llm_result.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/qwen/Qwen2.5-Omni-7B",
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

    model, processor = load_model(args.model_path)
    process_file(model, processor, input_path, output_path)


if __name__ == "__main__":
    main()
