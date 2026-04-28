from __future__ import annotations
import json
import torch
from src.utils.prompt_templates import LLM_CAUSAL_INFERENCE, LLM_STORY_GENERATION


def _load_llm(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto"
    )
    return model, tokenizer


def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move to GPU if the inputs object supports it (real BatchEncoding does; plain dicts don't)
    if hasattr(inputs, "to"):
        inputs = inputs.to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output[0][input_len:], skip_special_tokens=True
    ).strip()


def infer_causal_narrative(
    descriptions: dict[str, str], context: str, model_name: str
) -> str:
    """Step 1: Ask LLM to describe the narrative arc across all photos."""
    k = len(descriptions)
    desc_block = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions.values()))
    prompt = LLM_CAUSAL_INFERENCE.format(k=k, context=context, descriptions=desc_block)
    model, tokenizer = _load_llm(model_name)
    narrative = _generate_text(model, tokenizer, prompt, max_new_tokens=256)
    del model
    torch.cuda.empty_cache()
    return narrative


def generate_story(
    descriptions: dict[str, str],
    narrative: str,
    context: str,
    style: str,
    model_name: str,
) -> list[str]:
    """Step 2: Generate k-page story as a JSON list."""
    k = len(descriptions)
    desc_block = "\n".join(
        f"Page {i+1}: {desc}" for i, desc in enumerate(descriptions.values())
    )
    prompt = LLM_STORY_GENERATION.format(
        k=k, style=style, context=context, narrative=narrative, descriptions=desc_block
    )
    model, tokenizer = _load_llm(model_name)
    raw = _generate_text(model, tokenizer, prompt, max_new_tokens=512)
    del model
    torch.cuda.empty_cache()

    try:
        pages = json.loads(raw)
        if isinstance(pages, list) and len(pages) == k:
            return [str(p) for p in pages]
    except (json.JSONDecodeError, ValueError):
        pass

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    while len(lines) < k:
        lines.append("")
    return lines[:k]


def run_stage1(stage0_result: dict, context: str, style: str, config: dict) -> dict:
    """
    Returns:
        {"pages": list[str], "narrative": str}
    """
    descriptions = stage0_result["descriptions"]
    model_name = config["model"]
    narrative = ""

    if config["use_causal_inference"] and context:
        narrative = infer_causal_narrative(descriptions, context, model_name)

    pages = generate_story(descriptions, narrative, context, style, model_name)
    return {"pages": pages, "narrative": narrative}
