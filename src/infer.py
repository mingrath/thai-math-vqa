"""
vLLM self-consistency inference for Thai Math VQA.

For each image we sample N reasoning paths (temperature>0), extract the boxed
final answer from each, then majority-vote on the NORMALIZED answer and emit the
most common raw representative of the winning cluster.

Default model fits Colab Pro (A100 40GB / L4 24GB). Swap MODEL via --model.
"""
import os
import time
import pandas as pd

from .prompts import build_messages
from .normalize import extract_answer, majority_vote

# Default factor-28 pixel budget for Qwen2.5-VL (raise max for OCR-heavy Thai math,
# keep modest so KV cache fits a 24-40GB Colab GPU).
DEFAULT_MIN_PIXELS = 256 * 28 * 28   # ~200k px
DEFAULT_MAX_PIXELS = 1280 * 28 * 28  # ~1.0M px


def build_llm(model: str, max_model_len: int = 8192, tp: int = 1,
              gpu_mem_util: float = 0.92, max_pixels: int = DEFAULT_MAX_PIXELS):
    """Create a vLLM engine configured for single-image multimodal input."""
    from vllm import LLM
    return LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": DEFAULT_MIN_PIXELS, "max_pixels": max_pixels},
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=False,
        seed=0,
    )


def _make_request(processor, image_path, cot, min_pixels, max_pixels):
    """Return a vLLM request dict: {'prompt': str, 'multi_modal_data': {'image': ...}}."""
    from qwen_vl_utils import process_vision_info
    messages = build_messages(image_path, cot=cot,
                              min_pixels=min_pixels, max_pixels=max_pixels)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    return {"prompt": prompt, "multi_modal_data": {"image": image_inputs}}


def run_inference(df: pd.DataFrame, model: str, *, n: int = 8, temperature: float = 0.7,
                  top_p: float = 0.9, max_tokens: int = 1024, cot: bool = True,
                  max_pixels: int = DEFAULT_MAX_PIXELS, tp: int = 1,
                  max_model_len: int = 8192, gpu_mem_util: float = 0.92,
                  llm=None, processor=None, return_samples: bool = False):
    """
    Run self-consistency inference over df (must have columns: id, abs_path).
    Returns a DataFrame with columns: id, answer [, n_votes, n_valid, samples].
    `llm`/`processor` can be passed in to reuse a loaded engine across calls.
    """
    from transformers import AutoProcessor
    from vllm import SamplingParams

    if processor is None:
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    if llm is None:
        llm = build_llm(model, max_model_len=max_model_len, tp=tp,
                        gpu_mem_util=gpu_mem_util, max_pixels=max_pixels)

    # greedy single-sample when n==1, else diverse sampling for voting
    if n <= 1:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens)
    else:
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                            max_tokens=max_tokens, seed=0)

    requests = [
        _make_request(processor, p, cot, DEFAULT_MIN_PIXELS, max_pixels)
        for p in df["abs_path"].tolist()
    ]

    t0 = time.time()
    outputs = llm.generate(requests, sampling_params=sp)
    dt = time.time() - t0
    print(f"[infer] {len(df)} images x n={n} in {dt:.1f}s ({dt/max(len(df),1):.2f}s/img)")

    rows = []
    for rid, out in zip(df["id"].tolist(), outputs):
        raw_texts = [o.text for o in out.outputs]
        extracted = [extract_answer(t) for t in raw_texts]
        best_raw, votes, valid = majority_vote(extracted)
        row = {"id": rid, "answer": best_raw, "n_votes": votes, "n_valid": valid}
        if return_samples:
            row["samples"] = extracted
            row["raw"] = raw_texts
        rows.append(row)

    return pd.DataFrame(rows), llm, processor


def write_submission(pred_df: pd.DataFrame, out_path: str):
    """Write a Kaggle submission: columns id,answer (preserving the order of pred_df)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub = pred_df[["id", "answer"]].copy()
    sub["answer"] = sub["answer"].fillna("").astype(str)
    # guard: never submit an empty answer (grader may reject); fall back to "0"
    sub.loc[sub["answer"].str.strip() == "", "answer"] = "0"
    sub.to_csv(out_path, index=False)
    print(f"[submission] wrote {len(sub)} rows -> {out_path}")
    return out_path
