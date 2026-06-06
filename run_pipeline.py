#!/usr/bin/env python3
"""
End-to-end Thai Math VQA pipeline entrypoint.

Modes:
  eval     -- run on train split (has gold answers), print exact-match accuracy
  predict  -- run on test split, write submissions/sub.csv

Examples:
  # quick smoke test on 16 train images, greedy
  python run_pipeline.py --mode eval --limit 16 --n 1

  # full self-consistency eval on train (measure expected score)
  python run_pipeline.py --mode eval --n 8

  # produce the submission
  python run_pipeline.py --mode predict --n 8 --out submissions/sub.csv
"""
import argparse
import sys

from src.data import load_split
from src.infer import run_inference, write_submission, DEFAULT_MAX_PIXELS
from src.normalize import equiv


def main():
    ap = argparse.ArgumentParser(description="Thai Math VQA pipeline")
    ap.add_argument("--mode", choices=["eval", "predict"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HF model id. Fits Colab Pro: Qwen2.5-VL-7B / Qwen3-VL-8B. "
                         "Bigger (32B) needs A100 80GB or AWQ quant.")
    ap.add_argument("--n", type=int, default=8, help="self-consistency samples (1=greedy)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                    help="image token budget (px). Lower if you OOM.")
    ap.add_argument("--no-cot", action="store_true", help="answer-only prompt (faster)")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                    help="inference engine: vllm (fast), hf (transformers, robust on "
                         "Colab if vLLM/CUDA breaks), auto (try vllm then fall back)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (#GPUs)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=0, help="only first K rows (0=all)")
    ap.add_argument("--out", default="submissions/sub.csv")
    ap.add_argument("--save-eval", default="", help="optional csv to dump eval predictions")
    args = ap.parse_args()

    split = "train" if args.mode == "eval" else "test"
    df = load_split(args.data_dir, split)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"[data] {split}: {len(df)} rows")

    pred_df, _, _ = run_inference(
        df, args.model, n=args.n, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, cot=not args.no_cot, max_pixels=args.max_pixels,
        tp=args.tp, max_model_len=args.max_model_len, gpu_mem_util=args.gpu_mem_util,
        backend=args.backend,
    )

    if args.mode == "eval":
        merged = df[["id", "answer"]].merge(
            pred_df.rename(columns={"answer": "pred"}), on="id", how="left"
        )
        merged["correct"] = merged.apply(lambda r: equiv(r["pred"], r["answer"]), axis=1)
        acc = merged["correct"].mean()
        print(f"\n[eval] exact-match accuracy: {acc:.4f}  ({merged['correct'].sum()}/{len(merged)})")
        # show a few misses to guide prompt/normalize tuning
        miss = merged[~merged["correct"]].head(15)
        if len(miss):
            print("\n[eval] sample misses (gold | pred):")
            for _, r in miss.iterrows():
                print(f"  id={r['id']:>4}  {str(r['answer'])!r:28} | {str(r['pred'])!r}")
        if args.save_eval:
            merged.to_csv(args.save_eval, index=False)
            print(f"[eval] dumped predictions -> {args.save_eval}")
    else:
        write_submission(pred_df, args.out)
        print("\n[predict] done. Submit with:")
        print(f"  kaggle competitions submit -c "
              f"super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen "
              f"-f {args.out} -m 'qwen2.5-vl-7b sc-n{args.n}'")


if __name__ == "__main__":
    sys.exit(main())
