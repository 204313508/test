"""prompt_to_image.py

Convert user prompt text from dataset JSONL files into image files (white background, black
text). A new JSONL with an extra field ``image_path`` is created alongside the
original dataset.

Example
-------
python prompt_to_image.py --input data/ceval/test.jsonl --font_path "C:/Windows/Fonts/simhei.ttf"

Dependencies
------------
Pillow >= 10.0.0
If you need to support LaTeX style math formulas with proper rendering, install
matplotlib and pass ``--use_matplotlib``.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
import os
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------- Helper Functions ----------------------------- #

def build_user_prompt(record: Dict[str, str], chinese: bool = True) -> str:
    """Generate the *user* side prompt string based on the CEval/MMLU record.

    This mirrors the logic in ``build_prompt`` from ``test.py`` but returns only
    the user content (question + options), excluding system instructions.
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


def load_font(font_path: Optional[str] = None, font_size: int = 28) -> ImageFont.FreeTypeFont:  # type: ignore
    """Load a TTF font. Use SimSun (宋体) if available in the script directory.

    Parameters
    ----------
    font_path : Optional[str]
        Path to a TTF/TTC font. If ``None``, will attempt to use
        ``SimSun.ttc`` next to this script. Fallback to PIL default.
    font_size : int
        Font size in pixels.
    """
    if font_path is None:
        script_dir = Path(__file__).resolve().parent
        font_path = script_dir / "SimSun.ttc"

    try:
        if font_path and Path(font_path).exists():
            return ImageFont.truetype(str(font_path), font_size)
    except Exception:
        pass  # fallback to default
    return ImageFont.load_default()


def text_to_image(
    text: str,
    font: ImageFont.FreeTypeFont,  # type: ignore
    padding: int = 20,
    line_spacing: int = 10,
    max_width: int = 1200,
) -> Image.Image:
    """Render text to an image with white background.

    The function performs naive line wrapping based on ``max_width``.
    """
    # Wrap text so that each line fits within max_width
    draw_dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    def _text_wh(txt: str) -> Tuple[int, int]:
        """Return width and height of *txt* using current font."""
        bbox = draw_dummy.textbbox((0, 0), txt, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    wrapped_lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            wrapped_lines.append("")
            continue
        # Use binary search to find optimal wrap width per paragraph
        words = list(paragraph)
        # When dealing with Chinese characters, splitting by character instead of space
        if " " not in paragraph:
            # Simple per-char wrapping
            cur_line = ""
            for ch in words:
                w, _ = _text_wh(cur_line + ch)
                if w > max_width - 2 * padding:
                    wrapped_lines.append(cur_line)
                    cur_line = ch
                else:
                    cur_line += ch
            wrapped_lines.append(cur_line)
        else:
            # For space-separated languages
            cur_line = ""
            for word in paragraph.split(" "):
                candidate = word if cur_line == "" else f"{cur_line} {word}"
                w, _ = _text_wh(candidate)
                if w > max_width - 2 * padding:
                    wrapped_lines.append(cur_line)
                    cur_line = word
                else:
                    cur_line = candidate
            wrapped_lines.append(cur_line)

    line_heights = [_text_wh(line)[1] for line in wrapped_lines]
    text_height = sum(line_heights) + line_spacing * (len(wrapped_lines) - 1)
    img_height = text_height + 2 * padding
    img_width = max_width

    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    y = padding
    for idx, line in enumerate(wrapped_lines):
        draw.text((padding, y), line, fill="black", font=font)
        y += line_heights[idx] + line_spacing
    return img


# ---------------------------- Core Processing ------------------------------ #

def is_image_valid(path: Path) -> bool:
    """Return ``True`` if *path* exists and looks like a complete image file.
    
    We check file existence and that its size is larger than a minimum threshold.
    This is not a fool-proof validation but catches the vast majority of 
    truncated / empty files caused by unexpected termination.
    """
    MIN_IMAGE_SIZE_BYTES = 1024  # simple heuristic
    try:
        return path.exists() and path.stat().st_size >= MIN_IMAGE_SIZE_BYTES
    except Exception:
        return False


def process_file(
    input_jsonl: Path,
    font_path: Optional[str] = None,
    max_width: int = 1200,
    chinese: bool = True,
):
    """Convert *input_jsonl* prompts to images with breakpoint-resume support.
    
    The routine works in two phases:
    1. Analyse existing image files and determine which indices need to (re)generate.
       - Missing files are always regenerated.
       - The ten largest indices (latest 10) are always regenerated to guard
         against a previous abrupt stop that could have produced truncated files.
    2. Iterate over records that require generation.
    
    After all image files are ensured to be present and valid, a fresh JSONL
    with the additional ``image_path`` field is written next to the dataset.
    """
    dataset_dir = input_jsonl.parent  # e.g. data/ceval
    dataset_name = dataset_dir.name

    image_dir = dataset_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = dataset_dir / f"{dataset_name}_image.jsonl"

    font = load_font(font_path)
    
    # Load all records first – we will rewrite JSONL at the end
    with input_jsonl.open("r", encoding="utf-8") as fin:
        lines = [ln.strip() for ln in fin if ln.strip()]
    total_records = len(lines)
    
    # Determine indices to generate
    need_generate: List[int] = []
    last_10_indices = set(range(max(1, total_records - 9), total_records + 1))
    
    for idx in range(1, total_records + 1):
        image_path = image_dir / f"{idx:05d}.png"
        if (idx in last_10_indices) or (not is_image_valid(image_path)):
            need_generate.append(idx)
    
    if need_generate:
        print(f"(Re)Generating {len(need_generate)} image files out of {total_records} total")
    else:
        print("All image files present – validating")
    
    # Generate missing/corrupted images
    for idx in need_generate:
        record: Dict = json.loads(lines[idx - 1])
        prompt_text = build_user_prompt(record, chinese=chinese)
        
        img = text_to_image(prompt_text, font=font, max_width=max_width)
        img_rel_path = Path("image") / f"{idx:05d}.png"
        full_img_path = image_dir / img_rel_path.name
        img.save(full_img_path)
        
        if idx % 50 == 0 or idx in list(need_generate)[-10:]:
            print(f"Generated image {idx}/{total_records}")
    
    # Write (or rewrite) output JSONL with image_path field
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(lines, start=1):
            record: Dict = json.loads(line)
            img_rel_path = Path("image") / f"{idx:05d}.png"
            full_img_path = image_dir / img_rel_path.name
            # relative to current working directory so that downstream code can locate easily
            record["image_path"] = os.path.relpath(full_img_path, Path.cwd()).replace("\\", "/")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Finished. Images saved to", image_dir)
    print("New JSONL saved to", output_jsonl)


# --------------------------------- CLI ------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Convert dataset prompts to images")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Path to a TTF font file supporting Chinese characters",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=1200,
        help="Maximum width of the generated images in pixels",
    )
    parser.add_argument(
        "--chinese",
        type=int,
        default=1,
        help="Set to 1 for Chinese prompts, 0 for English prompts",
    )

    args = parser.parse_args()
    process_file(Path(args.input), font_path=args.font_path, max_width=args.max_width, chinese=(args.chinese==1))


if __name__ == "__main__":
    main()
