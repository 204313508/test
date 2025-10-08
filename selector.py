import argparse
import asyncio
import io
import json
import os
import pickle
import tempfile
import textwrap
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio  # type: ignore
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from transformers import (
    AutoProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

try:
    import edge_tts  # type: ignore
except Exception:
    edge_tts = None

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def free_gpu_memory():
    """Release cached GPU memory to mitigate out-of-memory errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_dataset(path: str) -> List[dict]:
    """Load JSONL dataset; returns list of dicts or empty list if path invalid."""
    if not path or not os.path.exists(path):
        return []
    
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_shared_prompt(question: str) -> str:
    """Build a shared prompt with concise modality-specific cues."""
    cues = "\n[Instruction] Provide a concise, factual answer."
    return f"{question.strip()}" + cues


def render_text_to_image(
    prompt: str,
    width: int = 1024,
    height: int = 768,
    font_path: Optional[str] = None,
    font_size: int = 28,
    max_line_width: int = 80,
) -> Image.Image:
    """Render prompt text onto a white canvas and return a PIL Image.
    Splits text into lines based on max_line_width using monospaced approximation.
    """
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    wrapped = []
    for para in prompt.split("\n"):
        wrapped.extend(textwrap.wrap(para, width=max_line_width) or [""])

    x, y = 40, 40
    line_height = int(font_size * 1.5)
    for line in wrapped:
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += line_height
        if y > height - line_height:
            break
    return image


async def _synthesize_tts_mp3(prompt: str, voice: str, rate: str, volume: str) -> bytes:
    if edge_tts is None:
        raise RuntimeError("edge_tts is not installed; cannot synthesize audio.")
    communicate = edge_tts.Communicate(prompt, voice=voice, rate=rate, volume=volume)
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk[0] == "audio":
            mp3_bytes += chunk[1]
    return mp3_bytes


def synthesize_tts_to_numpy(
    prompt: str,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    volume: str = "+0%",
    target_sr: int = 16000,
) -> np.ndarray:
    """Use edge_tts to synthesize prompt to MP3 in-memory, load with torchaudio,
    convert to mono numpy array at target_sr.
    """
    try:
        mp3_bytes = asyncio.get_event_loop().run_until_complete(
            _synthesize_tts_mp3(prompt, voice, rate, volume)
        )
    except RuntimeError as e:
        # edge_tts missing or other runtime issue
        print(f"TTS synthesis skipped: {e}")
        return np.array([], dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tmp.write(mp3_bytes)
        tmp.flush()
        waveform, sr = torchaudio.load(tmp.name)

    # to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).cpu().numpy().astype(np.float32)


def merge_datasets(text_data: List[dict], image_data: List[dict], audio_data: List[dict], image_dir: str, audio_dir: str) -> List[dict]:
    """
    Merge datasets by question, avoiding duplication while preserving modality-specific paths.
    
    Returns:
        List of merged samples, each containing:
        - question: str
        - expert: int (0 or 1)
        - modalities: dict with available modalities and their paths
    """
    question_map = {}
    
    # Process text data
    for item in text_data:
        question = item.get("question", "")
        if question:
            if question not in question_map:
                question_map[question] = {
                    "question": question,
                    "expert": item.get("expert", 0),
                    "modalities": {}
                }
            question_map[question]["modalities"]["text"] = {}
    
    # Process image data
    for idx, item in enumerate(image_data):
        question = item.get("question", "")
        if question:
            if question not in question_map:
                question_map[question] = {
                    "question": question,
                    "expert": item.get("expert", 0),
                    "modalities": {}
                }
            image_path = os.path.join(image_dir, f"{idx+1:05d}.png")
            question_map[question]["modalities"]["image"] = {
                "image_path": image_path
            }
    
    # Process audio data
    for idx, item in enumerate(audio_data):
        question = item.get("question", "")
        if question:
            if question not in question_map:
                question_map[question] = {
                    "question": question,
                    "expert": item.get("expert", 0),
                    "modalities": {}
                }
            audio_path = os.path.join(audio_dir, f"{idx+1:05d}.mp3")
            question_map[question]["modalities"]["audio"] = {
                "audio_path": audio_path
            }
    
    return list(question_map.values())

# ---------------------------------------------------------------------------
# Audio loading helper (borrowed from audio_infer.py)
# ---------------------------------------------------------------------------

def load_audio(path: str | os.PathLike, target_sr: int = 16000):
    """Load audio file and resample to *target_sr* mono tensor (numpy array)."""
    waveform, sr = torchaudio.load(str(path))  # (channels, time)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Qwen expects float32 numpy array
    return waveform.squeeze(0).cpu().numpy()



def prepare_inputs(merged_sample: dict, modality: str, processor):
    """Prepare processor inputs depending on modality; returns dict for model."""
    question = merged_sample.get("question", "")
    modality_data = merged_sample.get("modalities", {}).get(modality, {})
    
    if modality == "text":
        prompt = question
        inputs = processor(text=prompt, return_tensors="pt")
    
    elif modality == "image":
        image_path = modality_data.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f"Question: {question}\nPlease answer this question based on the image."
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    elif modality == "audio":
        audio_path = modality_data.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        try:
            # Load audio and prepare processor inputs (following audio_infer.py)
            audio_tensor = load_audio(audio_path)
            prompt = f"Question: {question}\nPlease answer this question based on the audio."
            inputs = processor(
                text=prompt,
                audio=[audio_tensor],  # pass list of np arrays
                return_tensors="pt",
                use_audio_in_video=True,
            )
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    else:
        return None
    
    return inputs


def prepare_intrinsic_inputs(
    question: str,
    modality: str,
    processor,
    font_path: Optional[str] = None,
    font_size: int = 28,
    canvas_width: int = 1024,
    canvas_height: int = 768,
    max_line_width: int = 80,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    volume: str = "+0%",
) -> Optional[dict]:
    """Prepare intrinsic (on-the-fly) inputs for image/audio from a text question."""
    prompt = build_shared_prompt(question)
    if modality == "text":
        return processor(text=prompt, return_tensors="pt")
    elif modality == "image":
        image = render_text_to_image(prompt, width=canvas_width, height=canvas_height,
                                     font_path=font_path, font_size=font_size, max_line_width=max_line_width)
        return processor(text=f"Question: {question}", images=image, return_tensors="pt")
    elif modality == "audio":
        audio_np = synthesize_tts_to_numpy(prompt, voice=voice, rate=rate, volume=volume)
        if audio_np.size == 0:
            return None
        return processor(text=f"Question: {question}", audio=[audio_np], return_tensors="pt", use_audio_in_video=True)
    else:
        return None


def extract_last_hidden(model, inputs, device):
    """Return last-token hidden state (np.ndarray shape (hidden_dim,))."""
    if inputs is None:
        return None
    
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            last_token_hidden = last_hidden_states[0, -1, :].cpu().numpy()  # (hidden_dim,)
        return last_token_hidden
    except Exception as e:
        print(f"Error extracting hidden states: {e}")
        return None


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def build_expert_regions(
    model_name: str,
    merged_data: List[dict],
    variance_threshold: float = 0.95,
    alpha: float = 2.0,
    font_path: Optional[str] = None,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    volume: str = "+0%",
) -> Tuple[PCA, Dict[str, np.ndarray], float]:
    """Fit PCA on expert hidden states with variance retention, compute centroids and text reliability threshold tau."""
    
    device = get_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    model.to(device)
    
    print(f"[Expert Regions] Using device: {device}")
    print(f"[Expert Regions] Model loaded: {model_name}")
    
    # Collect hidden states from expert samples
    expert_hiddens = {"text": [], "image": [], "audio": []}
    
    for sample in merged_data:
        if sample.get("expert") != 1:
            continue
        
        # Always collect text hidden from raw text
        q = sample.get("question", "")
        text_inputs = processor(text=build_shared_prompt(q), return_tensors="pt")
        text_hidden = extract_last_hidden(model, text_inputs, device)
        if text_hidden is not None:
            expert_hiddens["text"].append(text_hidden)

        # If dataset provides modalities, also collect; else optionally use intrinsic generation to enrich centroids
        available_modalities = list(sample.get("modalities", {}).keys())
        for modality in available_modalities:
            if modality == "text":
                continue
            inputs = prepare_inputs(sample, modality, processor)
            hidden = extract_last_hidden(model, inputs, device)
            if hidden is not None:
                expert_hiddens[modality].append(hidden)
        # Do NOT perform intrinsic augmentation; rely solely on provided dataset modalities
    
    # Clean up GPU memory after extraction
    model.to("cpu")
    del model, processor
    free_gpu_memory()
    
    # Convert to numpy arrays
    for modality in expert_hiddens:
        if expert_hiddens[modality]:
            expert_hiddens[modality] = np.array(expert_hiddens[modality])
        else:
            expert_hiddens[modality] = np.empty((0, 0))
    
    print(f"[Expert Regions] Collected expert hidden states:")
    for modality, hiddens in expert_hiddens.items():
        print(f"  {modality}: {hiddens.shape}")
    
    # Combine all expert hidden states for PCA fitting
    all_expert_hiddens = []
    for modality, hiddens in expert_hiddens.items():
        if hiddens.size > 0:
            all_expert_hiddens.append(hiddens)
    
    if not all_expert_hiddens:
        raise ValueError("No expert hidden states found!")
    
    combined_hiddens = np.vstack(all_expert_hiddens)
    print(f"[Expert Regions] Combined expert hiddens shape: {combined_hiddens.shape}")
    
    # Fit PCA to full dimension first, then choose k to hit variance threshold
    full_pca = PCA()
    full_pca.fit(combined_hiddens)
    cumvar = np.cumsum(full_pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = max(1, min(k, combined_hiddens.shape[1]))
    pca = PCA(n_components=k)
    pca.fit(combined_hiddens)
    print(f"[Expert Regions] PCA fitted with k={k} to retain >= {variance_threshold:.2f} variance")
    print(f"[Expert Regions] Retained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Compute centroids for each modality in PCA space
    centroids = {}
    text_distances = []
    for modality, hiddens in expert_hiddens.items():
        if hiddens.size > 0:
            projected = pca.transform(hiddens)
            centroids[modality] = np.mean(projected, axis=0)
            print(f"[Expert Regions] {modality} centroid computed from {len(hiddens)} samples")
            if modality == "text":
                # collect distances for tau
                dists = np.linalg.norm(projected - centroids["text"], axis=1)
                text_distances.extend(dists.tolist())

    # Reliability threshold tau = mu + alpha * sigma
    if text_distances:
        mu = float(np.mean(text_distances))
        sigma = float(np.std(text_distances))
        tau = mu + alpha * sigma
    else:
        tau = float("inf")
    print(f"[Expert Regions] Computed tau (alpha={alpha}): {tau:.6f}")

    return pca, centroids, tau


def run_selection_evaluation(
    merged_data: list,
    pca: PCA,
    centroids: dict,
    tau: float,
    model_name: str,
    progress_interval: int = 100,
    font_path: Optional[str] = None,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    volume: str = "+0%",
):
    """Run two-stage routing: reliability check on text, else intrinsic retrieval and reranking."""
    device = get_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    model.to(device)

    results = []
    for i, sample in enumerate(merged_data):
        if i % progress_interval == 0:
            print(f"[Selection] (n_components={pca.n_components_}) sample {i+1}/{len(merged_data)}")

        question = sample.get("question", "")

        modality_vectors = {}
        # Step 1: Reliability check using text only
        text_inputs = processor(text=build_shared_prompt(question), return_tensors="pt")
        text_hidden = extract_last_hidden(model, text_inputs, device)
        distances = {}
        selected_modality = None
        if text_hidden is not None:
            text_proj = pca.transform(text_hidden.reshape(1, -1))[0]
            d_text = float(np.linalg.norm(text_proj - centroids.get("text", text_proj)))
            distances["text"] = d_text
            if d_text < tau:
                selected_modality = "text"
                modality_vectors["text"] = text_proj
        
        # Step 2: If unreliable, use provided dataset modalities (image/audio) and rerank
        if selected_modality is None:
            # include text (if available)
            if text_hidden is not None:
                modality_vectors["text"] = text_proj
            # add available dataset modalities
            available_modalities = list(sample.get("modalities", {}).keys())
            for m in ("image", "audio"):
                if m in available_modalities:
                    inp = prepare_inputs(sample, m, processor)
                    hid = extract_last_hidden(model, inp, device)
                    if hid is not None:
                        proj = pca.transform(hid.reshape(1, -1))[0]
                        modality_vectors[m] = proj
                        distances[m] = float(np.linalg.norm(proj - centroids.get(m, proj)))

            if modality_vectors:
                selected_modality, distances = select_best_modality(modality_vectors, centroids)

            modality_results = {
                m: {
                    "vector_computed": True,
                    "distance_to_own_centroid": distances.get(m, None),
                }
                for m in modality_vectors
            }

            results.append({
                "question": question,
                "expert": sample.get("expert", 0),
                "available_modalities": list(modality_vectors.keys()),
                "modality_results": modality_results,
                "selection_result": {
                    "selected_modality": selected_modality,
                    "selection_method": "closest_to_own_centroid",
                    "distances_to_own_centroids": distances,
                },
            })

    # Compute accuracy (expert==1)
    correct = sum(1 for r in results if r["expert"] == 1)
    accuracy = correct / len(results) if results else 0.0

    # Clean-up GPU memory
    model.to("cpu")
    del model, processor
    free_gpu_memory()

    return results, accuracy


def select_best_modality(modality_vectors: Dict[str, np.ndarray], centroids: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, float]]:
    """
    Select the modality whose vector is closest to its corresponding expert centroid.
    
    Args:
        modality_vectors: Dict mapping modality names to their projected vectors
        centroids: Dict mapping modality names to their expert centroids
    
    Returns:
        Tuple of (selected_modality, distances_dict)
    """
    distances = {}
    
    # Calculate distance from each modality's vector to its corresponding centroid
    for modality, vector in modality_vectors.items():
        if modality in centroids:
            distance = float(np.linalg.norm(vector - centroids[modality]))
            distances[modality] = distance
    
    if not distances:
        # Fallback: if no centroids available, select first available modality
        selected_modality = next(iter(modality_vectors.keys()))
        return selected_modality, {}
    
    # Select modality with minimum distance to its own centroid
    selected_modality = min(distances, key=distances.get)
    
    return selected_modality, distances


def main():
    parser = argparse.ArgumentParser(description="IMRAG-based Dynamic Modality Selector Training")
    parser.add_argument("--model", required=True, help="Path to Qwen2-5-Omni model")
    parser.add_argument("--text_dataset", required=True, help="Path to text JSONL dataset")
    parser.add_argument("--image_dataset", required=False, default="", help="Path to image JSONL dataset (optional)")
    parser.add_argument("--audio_dataset", required=False, default="", help="Path to audio JSONL dataset (optional)")
    parser.add_argument("--image_dir", required=False, default="", help="Directory containing image files (optional)")
    parser.add_argument("--audio_dir", required=False, default="", help="Directory containing audio files (optional)")
    parser.add_argument("--output", required=True, help="Output path for selector artifact (.pkl)")
    parser.add_argument("--output_dataset", required=True, help="Output path for selection results (.jsonl)")
    parser.add_argument("--variance_threshold", type=float, default=0.95, help="PCA variance retention threshold (default 0.95)")
    parser.add_argument("--alpha", type=float, default=2.0, help="Reliability threshold alpha (mu + alpha*sigma)")
    parser.add_argument("--font_path", type=str, default="", help="Font path for text-to-image rendering")
    parser.add_argument("--voice", type=str, default="en-US-AriaNeural", help="edge_tts voice")
    parser.add_argument("--rate", type=str, default="+0%", help="edge_tts rate")
    parser.add_argument("--volume", type=str, default="+0%", help="edge_tts volume")

    args = parser.parse_args()

    # Load datasets
    text_data = load_dataset(args.text_dataset)
    image_data = load_dataset(args.image_dataset) if args.image_dataset else []
    audio_data = load_dataset(args.audio_dataset) if args.audio_dataset else []

    print(f"[Data Loading] Loaded datasets:")
    print(f"  Text: {len(text_data)} samples")
    print(f"  Image: {len(image_data)} samples")
    print(f"  Audio: {len(audio_data)} samples")

    # Merge datasets by question
    merged_data = merge_datasets(text_data, image_data, audio_data, args.image_dir, args.audio_dir)
    print(f"[Data Merging] Merged {len(merged_data)} unique questions from datasets.")

    # Build expert regions with PCA variance target and compute tau
    pca, centroids, tau = build_expert_regions(
        args.model,
        merged_data,
        variance_threshold=args.variance_threshold,
        alpha=args.alpha,
        font_path=args.font_path or None,
        voice=args.voice,
        rate=args.rate,
        volume=args.volume,
    )

    # Run two-stage selection
    results, accuracy = run_selection_evaluation(
        merged_data,
        pca,
        centroids,
        tau,
        args.model,
        progress_interval=100,
        font_path=args.font_path or None,
        voice=args.voice,
        rate=args.rate,
        volume=args.volume,
    )

    # Save selector artifact
    artifact = {
        "pca": pca,
        "centroids": centroids,
        "hidden_dim": int(pca.components_.shape[1]),
        "n_components": int(pca.n_components_),
        "accuracy": float(accuracy),
        "variance_threshold": float(args.variance_threshold),
        "alpha": float(args.alpha),
        "tau": float(tau),
    }
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(
        f"[Selector Training] Saved selector (k={pca.n_components_}, accuracy={accuracy:.4f}, tau={tau:.6f}) to {args.output}"
    )

    # Save selection results
    with open(args.output_dataset, "w", encoding="utf-8") as f_res:
        for r in results:
            f_res.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Selector Training] Saved {len(results)} selection results to {args.output_dataset}")

    # Print final accuracy summary and stats
    total = len(results)
    correct = sum(1 for r in results if r["expert"] == 1)
    overall_accuracy = correct / total if total > 0 else 0.0
    print(f"\n[Final Accuracy] {correct}/{total} = {overall_accuracy:.4f} (k={pca.n_components_})")

    modality_counts = {}
    expert_modality_counts = {}
    non_expert_modality_counts = {}
    for result in results:
        selected = result["selection_result"]["selected_modality"]
        is_expert = result["expert"] == 1
        modality_counts[selected] = modality_counts.get(selected, 0) + 1
        if is_expert:
            expert_modality_counts[selected] = expert_modality_counts.get(selected, 0) + 1
        else:
            non_expert_modality_counts[selected] = non_expert_modality_counts.get(selected, 0) + 1

    print(f"\n[Selection Statistics] Overall modality selection:")
    for modality, count in modality_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {modality}: {count} ({percentage:.1f}%)")

    print(f"\n[Selection Statistics] Expert samples modality selection:")
    total_expert = sum(expert_modality_counts.values())
    for modality, count in expert_modality_counts.items():
        percentage = (count / total_expert) * 100 if total_expert > 0 else 0
        print(f"  {modality}: {count} ({percentage:.1f}%)")

    print(f"\n[Selection Statistics] Non-expert samples modality selection:")
    total_non_expert = sum(non_expert_modality_counts.values())
    for modality, count in non_expert_modality_counts.items():
        percentage = (count / total_non_expert) * 100 if total_non_expert > 0 else 0
        print(f"  {modality}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
