from __future__ import annotations
import json
import random
import numpy as np
import piexif
from datetime import datetime
from PIL import Image
from sklearn.cluster import KMeans
from src.utils.prompt_templates import (
    VLM_DESCRIBE_PHOTO,
    LLM_SCORE_RELEVANCE,
)


def clip_cluster_select(
    image_paths: list[str],
    k: int,
    model_name: str,
    pretrained: str,
) -> list[str]:
    """Phase A: Select k diverse photos using CLIP embeddings + K-means."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()

    images = torch.stack([
        preprocess(Image.open(p).convert("RGB")) for p in image_paths
    ])

    with torch.no_grad():
        features = model.encode_image(images).cpu().numpy().astype(np.float32)

    features /= np.linalg.norm(features, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    selected: list[str] = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_feats = features[mask]
        center = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        local_best = int(np.argmin(dists))
        global_idx = int(np.where(mask)[0][local_best])
        selected.append(image_paths[global_idx])

    return selected


def describe_photos_with_vlm(image_paths: list[str], model_name: str) -> dict[str, str]:
    """Phase B step 1: Generate text description for each photo using VLM."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    descriptions: dict[str, str] = {}
    for path in image_paths:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": path},
            {"type": "text", "text": VLM_DESCRIBE_PHOTO},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[Image.open(path).convert("RGB")],
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
        generated = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        descriptions[path] = generated[0].strip()

    del model
    torch.cuda.empty_cache()
    return descriptions


def score_relevance_with_llm(
    descriptions: dict[str, str], context: str, model_name: str
) -> dict[str, float]:
    """Phase B step 2: Score each description's relevance to user context."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto"
    )

    scores: dict[str, float] = {}
    for path, desc in descriptions.items():
        prompt = LLM_SCORE_RELEVANCE.format(context=context, description=desc)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        raw = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            scores[path] = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            scores[path] = 0.5

    del model
    torch.cuda.empty_cache()
    return scores


def sort_by_exif(image_paths: list[str]) -> list[str]:
    """Phase C: Sort by EXIF DateTimeOriginal; images without EXIF go to end."""
    def get_dt(p: str) -> datetime | None:
        try:
            exif = piexif.load(p)
            raw = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            if raw:
                return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
        return None

    dated = [(p, get_dt(p)) for p in image_paths]
    with_dt = sorted([(p, dt) for p, dt in dated if dt], key=lambda x: x[1])
    without_dt = [p for p, dt in dated if not dt]
    return [p for p, _ in with_dt] + without_dt


def run_stage0(image_paths: list[str], context: str, config: dict) -> dict:
    """
    Full Stage 0: select K photos, describe them, order by time.

    Returns:
        {
            "selected_paths": list[str],    # k paths after selection
            "descriptions": dict[str, str], # path -> VLM description (ordered)
            "ordered_paths": list[str],     # final time-ordered k paths
        }
    """
    mode = config["mode"]
    k = config["k"]

    # Phase A: candidate selection
    if mode == "random":
        candidates = random.sample(image_paths, min(k, len(image_paths)))
    elif mode == "clip_only":
        candidates = clip_cluster_select(
            image_paths, k, config["clip_model"], config["clip_pretrained"]
        )
    elif mode in ("llm_only", "hybrid"):
        candidates = clip_cluster_select(
            image_paths, k, config["clip_model"], config["clip_pretrained"]
        )
    else:
        raise ValueError(f"Unknown stage0 mode: {mode}")

    # Phase B: describe all candidates with VLM
    descriptions = describe_photos_with_vlm(candidates, config["vlm_model"])

    # Phase B: score relevance against user context (llm_only and hybrid)
    if mode in ("llm_only", "hybrid") and context:
        scores = score_relevance_with_llm(descriptions, context, config["llm_model"])

        if mode == "llm_only":
            # Re-select top-k from all images by score
            all_descs = describe_photos_with_vlm(image_paths, config["vlm_model"])
            all_scores = score_relevance_with_llm(all_descs, context, config["llm_model"])
            sorted_paths = sorted(all_scores, key=lambda p: all_scores[p], reverse=True)
            candidates = sorted_paths[:k]
            descriptions = {p: all_descs[p] for p in candidates}

    # Phase C: order by EXIF timestamp
    ordered = sort_by_exif(candidates)
    ordered_descriptions = {p: descriptions.get(p, "") for p in ordered}

    return {
        "selected_paths": candidates,
        "descriptions": ordered_descriptions,
        "ordered_paths": ordered,
    }
