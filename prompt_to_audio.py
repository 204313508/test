"""prompt_to_audio.py

Convert user prompt text from dataset JSONL files into speech audio files using
Microsoft Edge text-to-speech (edge_tts). A new JSONL with an extra field
``audio_path`` is created alongside the original dataset.

Example
-------
python prompt_to_audio.py --input data/ceval/test.jsonl \
                         --voice zh-CN-XiaoxiaoNeural

Dependencies
------------
edge_tts  >= 6.1.10  # async wrapper around Microsoft Edge TTS endpoints

Install with::

    pip install edge-tts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional, List

import edge_tts  # type: ignore
from tqdm.auto import tqdm

# ---------------------------- Helper Functions ----------------------------- #

MIN_AUDIO_SIZE_BYTES = 1024  # simple heuristic to judge whether an mp3 file is complete

def is_audio_valid(path: Path) -> bool:
    """Return ``True`` if *path* exists and looks like a complete mp3.

    We simply check file existence and that its size is larger than
    ``MIN_AUDIO_SIZE_BYTES``. This is not a fool-proof validation but
    catches the vast majority of truncated / empty files caused by
    unexpected termination.
    """
    try:
        return path.exists() and path.stat().st_size >= MIN_AUDIO_SIZE_BYTES
    except Exception:
        return False

def build_user_prompt(record: Dict[str, str], chinese: bool = True) -> str:
    """Generate the *user* side prompt string based on a CEval/MMLU record.

    Mirrors the logic in ``build_user_prompt`` from ``prompt_to_image.py`` so
    the prompts remain identical across modalities.
    """
    if chinese:
        return (
            f"{record['question']}\n"
            f"选项为：A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "请直接给出答案的字母（A/B/C/D），不需要其他内容。"
        )
    else:
        return (
            f"{record['question']}\n"
            f"Options: A: {record['A']}\nB: {record['B']}\nC: {record['C']}\nD: {record['D']}\n"
            "Please output only the letter of the answer (A/B/C/D) with no other content."
        )


async def text_to_speech(
    text: str,
    out_path: Path,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    volume: str = "+0%",
):
    """Save *text* to *out_path* as mp3 using Edge TTS."""
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
    await communicate.save(str(out_path))


# ---------------------------- Core Processing ------------------------------ #

async def process_file_async(
    input_jsonl: Path,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    volume: str = "+0%",
    max_retries: int = 3,
    chinese: bool = True,
    workers: int = 5,
    output_dir: Optional[Path] = None,
):
    """Convert *input_jsonl* prompts to audio with breakpoint-resume support.

    The routine works in two phases:
    1. Analyse existing audio files and determine which indices need to (re)generate.
       - Missing files are always regenerated.
       - The ten largest indices (latest 10) are always regenerated to guard
         against a previous abrupt stop that could have produced truncated files.
    2. Iterate over records that require generation with retry and validation.

    After all audio files are ensured to be present and valid, a fresh JSONL
    with the additional ``audio_path`` field is written next to the dataset.
    """
    dataset_dir = input_jsonl.parent  # e.g. data/ceval
    dataset_name = dataset_dir.name

    # Use provided output_dir or default to dataset_dir/audio
    if 'output_dir' not in locals():
        output_dir = dataset_dir / "audio"
    audio_dir = Path(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = audio_dir / f"{dataset_name}_audio.jsonl"

    # Load all records first – we will rewrite JSONL at the end
    with input_jsonl.open("r", encoding="utf-8") as fin:
        lines = [ln.strip() for ln in fin if ln.strip()]
    total_records = len(lines)
    
    # Create output directory if it doesn't exist
    audio_dir = Path(output_dir) if 'output_dir' in locals() else dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Determine indices to generate
    need_generate: List[int] = []
    last_10_indices = set(range(max(1, total_records - 9), total_records + 1))

    for idx in range(1, total_records + 1):
        audio_path = audio_dir / f"{idx:05d}.mp3"
        if (idx in last_10_indices) or (not is_audio_valid(audio_path)):
            need_generate.append(idx)

    if need_generate:
        print(f"(Re)Generating {len(need_generate)} audio files out of {total_records} total")
        tqdm_desc = f"Generating audio files"
    else:
        print("All audio files present – validating")
        tqdm_desc = "Validating audio files"

    # Concurrency control
    semaphore = asyncio.Semaphore(workers)

    async def generate_one(idx: int):
        record: Dict = json.loads(lines[idx - 1])
        prompt_text = build_user_prompt(record, chinese=chinese)
        audio_path = audio_dir / f"{idx:05d}.mp3"
        async with semaphore:
            attempt = 0
            while True:
                attempt += 1
                try:
                    await text_to_speech(prompt_text, audio_path, voice, rate, volume)
                except Exception as exc:
                    print(f"[Warning] TTS error on record #{idx} attempt {attempt}: {exc}")
                if is_audio_valid(audio_path):
                    break  # success
                # cleanup and retry
                try:
                    audio_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                # Exponential backoff every 5 failures to avoid rapid hammering
                if attempt % 5 == 0:
                    await asyncio.sleep(3)
                else:
                    await asyncio.sleep(1)

    # Launch concurrent tasks and await completion with progress bar
    tasks = [asyncio.create_task(generate_one(idx)) for idx in need_generate]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=tqdm_desc):
        await fut

    # Write (or rewrite) output JSONL with audio_path field
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(lines, start=1):
            record: Dict = json.loads(line)
            audio_rel_path = Path("audio") / f"{idx:05d}.mp3"
            record["audio_path"] = os.path.relpath(audio_dir / audio_rel_path.name, Path.cwd()).replace("\\", "/")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Finished. Audio files saved to", audio_dir)
    print("JSONL with audio_path saved to", output_jsonl)


# --------------------------------- CLI ------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Convert dataset prompts to audio using edge_tts")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--voice",
        type=str,
        default="zh-CN-XiaoxiaoNeural",
        help="TTS voice name (see edge-tts voices list)",
    )
    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="Speaking rate, e.g. '+0%' or '-10%'",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="+0%",
        help="Volume, e.g. '+0%' or '+20%'",
    )
    parser.add_argument(
        "--chinese",
        type=int,
        default=1,
        help="Set to 1 for Chinese prompts, 0 for English prompts",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent TTS workers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save audio files (e.g., /tmp/audio20)",
    )

    args = parser.parse_args()

    # Auto-select voice if not specified differently
    if (args.chinese == 0) and args.voice.startswith("zh-"):
        args.voice = "en-US-AriaNeural"
    elif (args.chinese == 1) and args.voice.startswith("en-"):
        args.voice = "zh-CN-XiaoxiaoNeural"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(
        process_file_async(
            Path(args.input),
            voice=args.voice,
            rate=args.rate,
            volume=args.volume,
            chinese=(args.chinese==1),
            workers=args.workers,
            output_dir=output_dir
        )
    )


if __name__ == "__main__":
    main()
