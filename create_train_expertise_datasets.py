import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The original helper functions relied on an "expert subject summary" CSV to
# determine the categories where a modality excels.  The new requirement is to
# *derive* this information directly from `--accuracy_csv` by comparing the
# average accuracy of all modalities for each subject.  The modality with a
# strictly higher accuracy than the others is regarded as the expert (label
# 1), while the others are non-experts (label 0).  If there is a tie for the
# highest accuracy, *no* modality is considered expert for that subject.
# ---------------------------------------------------------------------------

def build_labels_from_accuracy(accuracy_csv: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:  # noqa: D401
    """Return (labels, scores) where
    labels: modality_cn -> {subject: expert_label}
    scores: modality_cn -> {subject: average_accuracy}
    """
    """Return a nested mapping modality_cn -> {subject: expert_label}."""
    df = pd.read_csv(accuracy_csv)

    # Ensure modality mapping EN->CN is consistent
    en2cn = {"audio": "语音", "text": "文本", "image": "图片"}

    # Build a dict of dicts to collect labels
    labels: Dict[str, Dict[str, int]] = {cn: {} for cn in en2cn.values()}
    scores: Dict[str, Dict[str, float]] = {cn: {} for cn in en2cn.values()}

    # Group by subject so we can compare across modalities
    tol = 1e-4  # Tolerance for considering two accuracies equal
    for subject, grp in df.groupby("subject"):
        # Build mapping modality_en -> accuracy for this subject
        acc_map = {row["type"]: row["average_accuracy"] for _, row in grp.iterrows()}

        if len(acc_map) < 3:
            # require all three modalities to appear to make a decision
            continue

        best_acc = max(acc_map.values())
        # Identify modalities achieving the best accuracy (within tolerance)
        best_mods = [m for m, a in acc_map.items() if abs(best_acc - a) < tol]

        # 只有当某个模态的准确率严格高于另外两个模态时，才将其标记为擅长；否则全部标记为非擅长。
        for modality_en, acc in acc_map.items():
            cn = en2cn[modality_en]
            # Store score for later inspection/debugging
            scores[cn][subject] = acc
            if len(best_mods) == 1 and modality_en in best_mods:
                labels[cn][subject] = 1  # expert
            else:
                labels[cn][subject] = 0  # non-expert

    return labels, scores

# The old heuristic for choosing non-expert subjects is no longer necessary
# because every subject will have an expert/non-expert label for every modality
# based on the comparison above.

def _deprecated_load_non_expert_subjects(*args, **kwargs):  # noqa: ANN001
    raise RuntimeError("load_non_expert_subjects is deprecated under the new accuracy-based labelling logic.")
    """Return the *max(expert_count, 3)* lowest-scoring subjects for each modality."""
    df = pd.read_csv(accuracy_csv)
    mapping: Dict[str, List[str]] = {}
    for modality_en, modality_cn in [("audio", "语音"), ("text", "文本"), ("image", "图片")]:
        sub_df = df[df["type"] == modality_en].sort_values("average_accuracy", ascending=True)
        n_expert = counts.get(modality_cn, 0)
        n = max(n_expert, 3)  # Ensure at least 3 non-expert subjects
        mapping[modality_cn] = sub_df["subject"].head(n).tolist()
    return mapping


def build_subject_labels(expert_map: Dict[str, List[str]], non_expert_map: Dict[str, List[str]], modality_cn: str):
    labels = {s: 1 for s in expert_map.get(modality_cn, [])}
    labels.update({s: 0 for s in non_expert_map.get(modality_cn, [])})
    return labels


def filter_and_write(jsonl_path: str, subject_labels: Dict[str, int], output_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            subj = obj.get("subject", "")
            if subj in subject_labels:
                obj["expert"] = subject_labels[subj]
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate modality-specific expert/non-expert datasets in JSONL format.")
    parser.add_argument("--data", required=True, help="Path to the text modality jsonl file (with question/answer pairs)")
    parser.add_argument("--audio_data", required=True, help="Path to the audio modality jsonl file (with audio_path field)")
    parser.add_argument("--image_data", required=True, help="Path to the image modality jsonl file (with image_path field)")
    parser.add_argument("--summary_csv", required=True, help="Path to <prefix>_expert_subject_summary.csv")
    parser.add_argument("--accuracy_csv", required=True, help="Path to combined_subject_accuracy.csv")
    parser.add_argument("--prefix", default="dataset", help="Prefix for output jsonl files")
    parser.add_argument("--suffix", required=True, help="Suffix to add to output jsonl files (e.g. version, tag)")
    args = parser.parse_args()

    # Build labels directly from accuracy comparison
    modality_labels, modality_scores = build_labels_from_accuracy(args.accuracy_csv)

    # 直接使用所有科目（不再进行采样筛选）
    # `modality_labels` 已经包含了每个模态下所有科目的专家(1)/非专家(0)标记
    # 如果之后有需要，可在此处添加其他处理逻辑
    # 注意：下面的循环将遍历每个模态并写出完整的训练集

    os.makedirs(os.path.dirname(os.path.abspath(args.prefix)), exist_ok=True)

    for modality_cn, modality_en in [("语音", "audio"), ("文本", "text"), ("图片", "image")]:
        labels = modality_labels.get(modality_cn, {})
        if not labels:
            print(f"跳过 {modality_cn}，未找到任何擅长/不擅长科目")
            continue
        # Select the correct input jsonl for the modality
        if modality_en == "audio":
            input_jsonl = args.audio_data
        elif modality_en == "image":
            input_jsonl = args.image_data
        else:
            input_jsonl = args.data

        out_path = f"{args.prefix}_train_expert_{modality_en}_{args.suffix}.jsonl"
        filter_and_write(input_jsonl, labels, out_path)
        print(f"已生成 {out_path} (共 {len(labels)} 个类别，包含擅长 {sum(v == 1 for v in labels.values())} / 不擅长 {sum(v == 0 for v in labels.values())})")


if __name__ == "__main__":
    main()
